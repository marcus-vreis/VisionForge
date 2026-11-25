/** Sending a dataset .zip to a shared server, in parts (ADR-246).
 *
 * The server gives the part size when the upload opens. The page slices the
 * file and sends one part after the other; a part that fails leaves the
 * upload open, and asking the server which parts arrived is how it resumes:
 * only the missing ones are sent again. The calls are passed in, so this runs
 * in the tests without a server.
 */

/** An open upload, as `POST /api/uploads` and `GET /api/uploads/{id}` answer. */
export interface DatasetUpload {
  id: string;
  /** The folder it becomes inside the person's dataset folder. */
  name: string;
  filename: string;
  size: number;
  part_size: number;
  parts: number;
  /** The parts that arrived, by number. */
  received: number[];
}

/** The dataset the server unpacked. */
export interface UploadedDataset {
  name: string;
  path: string;
  size_bytes: number;
}

export interface UploadCalls {
  sendPart: (id: string, part: number, data: Blob) => Promise<unknown>;
  complete: (id: string) => Promise<UploadedDataset>;
}

/** The folder a .zip becomes when the person names none: "beans.zip" → "beans". */
export function folderNameOf(fileName: string): string {
  return fileName.replace(/\.zip$/i, "").trim();
}

function partLength(upload: DatasetUpload, part: number): number {
  return Math.min(upload.part_size, upload.size - part * upload.part_size);
}

/** The parts the server does not have yet, in order. */
export function missingParts(upload: DatasetUpload): number[] {
  const arrived = new Set(upload.received);
  return Array.from({ length: upload.parts }, (_, part) => part).filter(
    (part) => !arrived.has(part),
  );
}

/** The bytes already on the server. */
export function receivedBytes(upload: DatasetUpload): number {
  return upload.received
    .filter((part) => part >= 0 && part < upload.parts)
    .reduce((sum, part) => sum + partLength(upload, part), 0);
}

/** Whole percent of `sent` out of `total`; an empty total reads as done. */
export function percentOf(sent: number, total: number): number {
  return total > 0 ? Math.min(100, Math.floor((sent / total) * 100)) : 100;
}

/** Send what the server is missing, one part at a time, then ask it to unpack.
 *
 * `onProgress` gets the bytes on the server after each part; `onFinishing`
 * runs once every part is there, before the unpacking, which may take a while.
 * A failure rejects with the call's error and leaves the upload open.
 */
export async function sendMissing(
  calls: UploadCalls,
  file: Blob,
  upload: DatasetUpload,
  onProgress: (sentBytes: number) => void,
  onFinishing: () => void = () => {},
): Promise<UploadedDataset> {
  if (file.size !== upload.size) {
    throw new Error(`the file has ${file.size} bytes and the upload ${upload.size}`);
  }
  let sent = receivedBytes(upload);
  onProgress(sent);
  for (const part of missingParts(upload)) {
    const start = part * upload.part_size;
    const length = partLength(upload, part);
    await calls.sendPart(upload.id, part, file.slice(start, start + length));
    sent += length;
    onProgress(sent);
  }
  onFinishing();
  return calls.complete(upload.id);
}
