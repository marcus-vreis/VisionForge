import { describe, expect, it } from "vitest";

import {
  folderNameOf,
  missingParts,
  percentOf,
  receivedBytes,
  sendMissing,
  type DatasetUpload,
  type UploadCalls,
} from "./dataset-upload";

/** 10 bytes in parts of 4: 4 + 4 + 2. */
const FILE = new Blob([new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]);
const OPEN: DatasetUpload = {
  id: "a".repeat(32),
  name: "beans",
  filename: "beans.zip",
  size: 10,
  part_size: 4,
  parts: 3,
  received: [],
};
const DONE = { name: "beans", path: "/srv/ana/datasets/beans", size_bytes: 10 };

/** A server that keeps what it receives, and fails a part when told to. */
function fakeServer(failOn: number | null = null) {
  const stored = new Map<number, number[]>();
  const order: number[] = [];
  let completed = 0;
  let failing = failOn;
  const calls: UploadCalls = {
    sendPart: async (_id, part, data) => {
      if (part === failing) {
        failing = null;
        throw new Error("connection lost");
      }
      order.push(part);
      stored.set(part, Array.from(new Uint8Array(await data.arrayBuffer())));
    },
    complete: async () => {
      completed += 1;
      return DONE;
    },
  };
  return {
    calls,
    order,
    stored,
    completed: () => completed,
    status: (): DatasetUpload => ({
      ...OPEN,
      received: [...stored.keys()].sort((a, b) => a - b),
    }),
  };
}

describe("the upload of a dataset .zip in parts (ADR-246)", () => {
  it("names the folder after the file when the person names none", () => {
    expect(folderNameOf("beans.zip")).toBe("beans");
    expect(folderNameOf("Beans.ZIP")).toBe("Beans");
    expect(folderNameOf("my data.v2.zip")).toBe("my data.v2");
  });

  it("knows which parts and how many bytes the server already has", () => {
    expect(missingParts(OPEN)).toEqual([0, 1, 2]);
    expect(missingParts({ ...OPEN, received: [2, 0] })).toEqual([1]);
    expect(receivedBytes({ ...OPEN, received: [2, 0] })).toBe(6);
    // A number outside the upload counts for nothing.
    expect(receivedBytes({ ...OPEN, received: [7] })).toBe(0);
  });

  it("reads progress as whole percent", () => {
    expect(percentOf(0, 10)).toBe(0);
    expect(percentOf(6, 10)).toBe(60);
    expect(percentOf(9.99, 10)).toBe(99);
    expect(percentOf(10, 10)).toBe(100);
    expect(percentOf(0, 0)).toBe(100);
  });

  it("sends each part's slice in order, then asks the server to unpack", async () => {
    const server = fakeServer();
    const progress: number[] = [];
    let finishing = 0;

    const dataset = await sendMissing(server.calls, FILE, OPEN, (b) => progress.push(b), () => {
      finishing += 1;
    });

    expect(dataset).toEqual(DONE);
    expect(server.order).toEqual([0, 1, 2]);
    expect(server.stored.get(0)).toEqual([0, 1, 2, 3]);
    expect(server.stored.get(2)).toEqual([8, 9]);
    expect(progress).toEqual([0, 4, 8, 10]);
    expect(finishing).toBe(1);
    expect(server.completed()).toBe(1);
  });

  it("resumes after a failure by sending only the missing parts", async () => {
    const server = fakeServer(1);
    await expect(sendMissing(server.calls, FILE, OPEN, () => {})).rejects.toThrow(
      "connection lost",
    );
    expect(server.completed()).toBe(0);
    expect(server.status().received).toEqual([0]);

    const progress: number[] = [];
    await sendMissing(server.calls, FILE, server.status(), (b) => progress.push(b));

    expect(server.order).toEqual([0, 1, 2]);
    expect(progress).toEqual([4, 8, 10]);
    expect(server.completed()).toBe(1);
  });

  it("refuses to resume with a file of another size", async () => {
    const server = fakeServer();
    const other = new Blob([new Uint8Array(11)]);
    await expect(sendMissing(server.calls, other, OPEN, () => {})).rejects.toThrow("11 bytes");
    expect(server.order).toEqual([]);
  });
});
