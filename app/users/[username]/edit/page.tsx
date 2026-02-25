import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { notFound, redirect } from "next/navigation";
import { query } from "@/lib/graphql/apollo-client";
import { GET_USER } from "@/lib/models/users/gql";
import { Page } from "@/components/page";
import { Container } from "@/components/container";
import { EditProfileForm } from "./edit-profile-form";
import type { GetUserQuery, GetUserQueryVariables } from "@/lib/graphql/__generated__/graphql";

export default async function EditProfilePage({
    params,
}: {
    params: Promise<{ username: string }>;
}) {
    const { username } = await params;

    const [session, { data }] = await Promise.all([
        auth.api.getSession({ headers: await headers() }),
        query<GetUserQuery, GetUserQueryVariables>({
            query: GET_USER,
            variables: { username },
        }),
    ]);

    const user = data?.user;
    if (!user) notFound();

    const sessionUsername = (session?.user as Record<string, unknown>)?.username as string | undefined;
    if (!sessionUsername || sessionUsername !== username) {
        redirect(`/users/${username}`);
    }

    return (
        <Page breadcrumbs={[
            { label: "Profilo", href: `/users/${username}` },
            { label: "Modifica" }
        ]}>
            <Container className="pb-16 space-y-8">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Modifica il tuo profilo</h1>
                    <p className="text-muted-foreground mt-2">Aggiorna le tue informazioni, caratteristiche e interessi.</p>
                </div>
                <EditProfileForm user={user} />
            </Container>
        </Page>
    );
}
