// Shared TypeSafe client for all sample apps. Key from env or ~/.hermes/jev/key.
import { TypeSafeClient } from "@typesafe-ai/sdk";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";

export function getKey() {
  if (process.env.TYPESAFE_API_KEY) return process.env.TYPESAFE_API_KEY;
  const f = path.join(os.homedir(), ".hermes/jev/key");
  if (fs.existsSync(f)) return fs.readFileSync(f, "utf8").trim();
  throw new Error("No TYPESAFE_API_KEY env var and no ~/.hermes/jev/key file");
}

export const client = new TypeSafeClient({ apiKey: getKey() });
