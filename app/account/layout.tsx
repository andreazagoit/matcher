import { auth } from "@/lib/oauth/auth";
import { redirect } from "next/navigation";

export default async function AccountLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const session = await auth();

    if (!session) {
        redirect("/api/auth/signin?callbackUrl=/account");
    }

    return <>{children}</>;
}
