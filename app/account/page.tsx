import { auth } from "@/lib/oauth/auth";
import { query } from "@/lib/graphql/apollo-client";
import { GET_ME } from "@/lib/models/users/gql";
import { AccountForm } from "./account-form";
import { AccountHeaderActions } from "./account-header-actions";
import { Page } from "@/components/page";
import { redirect } from "next/navigation";
import type { GetMeQuery } from "@/lib/graphql/__generated__/graphql";

export default async function AccountPage() {
    const session = await auth();

    if (!session?.user?.id) {
        redirect("/");
    }

    const { data } = await query<GetMeQuery>({ query: GET_ME });
    const user = data?.me;

    if (!user) {
        return <div>User not found</div>;
    }

    return (
        <Page
            breadcrumbs={[
                { label: "Account" }
            ]}
            header={
                <div className="space-y-1">
                    <h1 className="text-4xl font-extrabold tracking-tight text-foreground bg-clip-text">Account Settings</h1>
                    <p className="text-lg text-muted-foreground font-medium">Manage your personal information and profile preferences</p>
                </div>
            }
            actions={<AccountHeaderActions />}
        >
            <div className="w-full mx-auto space-y-8">
                <AccountForm initialUser={user} />
            </div>
        </Page>
    );
}
