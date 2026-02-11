import { auth } from "@/lib/oauth/auth";
import { getUserById } from "@/lib/models/users/operations";
import { AccountForm } from "./account-form";
import { AccountHeaderActions } from "./account-header-actions";
import { PageShell } from "@/components/page-shell";
import { redirect } from "next/navigation";

export default async function AccountPage() {
    const session = await auth();

    if (!session?.user?.id) {
        // Should be handled by layout, but for safety
        redirect("/");
    }

    const user = await getUserById(session.user.id);

    if (!user) {
        // Handle edge case where session exists but user DB record doesn't
        return <div>User not found</div>;
    }

    // Map DB user to UserData interface expected by form
    const userData = {
        id: user.id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        birthDate: user.birthDate ? new Date(user.birthDate).toISOString().split('T')[0] : "", // Convert Date to string
        gender: (user.gender === "man" || user.gender === "woman" || user.gender === "non_binary") ? user.gender : null,
    };

    return (
        <PageShell
            header={
                <div className="space-y-1">
                    <h1 className="text-4xl font-extrabold tracking-tight text-foreground bg-clip-text">Account Settings</h1>
                    <p className="text-lg text-muted-foreground font-medium">Manage your personal information and profile preferences</p>
                </div>
            }
            actions={<AccountHeaderActions />}
        >
            <div className="w-full mx-auto space-y-8">
                <AccountForm initialUser={userData} />
            </div>
        </PageShell>
    );
}


