import type { Metadata } from "next";
import { ModelView } from "@/components/ModelView";

export const metadata: Metadata = { title: "Model" };

export default function ModelPage() {
  return <ModelView />;
}
