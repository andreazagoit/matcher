import { auth } from "@/lib/oauth/auth";
import { query } from "@/lib/graphql/apollo-client";
import { AccountForm } from "./account-form";
import { AccountHeaderActions } from "./account-header-actions";
import { PageShell } from "@/components/page-shell";
import { redirect } from "next/navigation";
import gql from "graphql-tag";

const GET_ME = gql`
  query GetMe {
    me {
      id
      firstName
      lastName
      email
      birthDate
      gender
    }
  }
`;

interface UserData {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    birthDate: string;
    gender: "man" | "woman" | "non_binary" | null;
}

export default async function AccountPage() {
    const session = await auth();

    if (!session?.user?.id) {
        redirect("/");
    }

    const { data } = await query({ query: GET_ME });
    const user = (data as { me: UserData }).me;

    if (!user) {
        return <div>User not found</div>;
    }

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
                <AccountForm initialUser={user} />
            </div>
        </PageShell>
    );
}
