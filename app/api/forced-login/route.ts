import { signIn } from "@/lib/auth";

export async function GET() {
    await signIn("matcher", { redirectTo: "/dashboard" });
}
