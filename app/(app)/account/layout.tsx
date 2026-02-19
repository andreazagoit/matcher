import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { redirect } from "next/navigation";

export default async function AccountLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const session = await auth.api.getSession({
        headers: await headers(),
    });

    console.log("[DEBUG] AccountLayout: session is", session);

    if (!session) {
        redirect("/");
    }

    return <>{children}</>;
}
