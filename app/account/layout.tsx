import { auth, signIn } from "@/lib/oauth/auth";

export default async function AccountLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const session = await auth();

    if (!session) {
        await signIn("matcher", { redirectTo: "/account" });
    }

    return <>{children}</>;
}
