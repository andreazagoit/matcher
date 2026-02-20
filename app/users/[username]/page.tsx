import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { notFound } from "next/navigation";
import { query } from "@/lib/graphql/apollo-client";
import { GET_USER } from "@/lib/models/users/gql";
import { Page } from "@/components/page";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ProfileSettings } from "./profile-settings";
import type { GetUserQuery, GetUserQueryVariables } from "@/lib/graphql/__generated__/graphql";
import { TAG_CATEGORIES } from "@/lib/models/tags/data";

const GENDER_LABEL: Record<string, string> = {
    man: "Man",
    woman: "Woman",
    non_binary: "Non-binary",
};

const TAG_TO_CATEGORY: Record<string, string> = {};
for (const [category, tags] of Object.entries(TAG_CATEGORIES)) {
    for (const tag of tags) TAG_TO_CATEGORY[tag] = category;
}

const CATEGORY_COLORS: Record<string, string> = {
    outdoor: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
    culture: "bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300",
    food: "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300",
    sports: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",
    creative: "bg-pink-100 text-pink-800 dark:bg-pink-900/40 dark:text-pink-300",
    social: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300",
};

function getAge(birthdate: string): number {
    const today = new Date();
    const birth = new Date(birthdate);
    let age = today.getFullYear() - birth.getFullYear();
    const m = today.getMonth() - birth.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
    return age;
}

export default async function UserProfilePage({
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
    const isOwnProfile = !!sessionUsername && sessionUsername === username;

    const age = user.birthdate ? getAge(user.birthdate) : null;
    const initials = `${user.givenName?.[0] ?? ""}${user.familyName?.[0] ?? ""}`.toUpperCase();

    return (
        <Page
            breadcrumbs={[
                { label: isOwnProfile ? "My Profile" : `${user.givenName} ${user.familyName}` },
            ]}
            header={
                <div className="flex flex-col sm:flex-row items-start sm:items-end gap-6">
                    <Avatar className="h-24 w-24 rounded-2xl border-4 border-background shadow-lg">
                        <AvatarImage src={user.image ?? undefined} alt={`${user.givenName} ${user.familyName}`} />
                        <AvatarFallback className="rounded-2xl text-2xl font-bold">{initials}</AvatarFallback>
                    </Avatar>
                    <div className="space-y-1.5">
                        <div className="flex items-center gap-3 flex-wrap">
                            <h1 className="text-4xl font-extrabold tracking-tight">
                                {user.givenName} {user.familyName}
                                {age !== null && (
                                    <span className="text-muted-foreground font-medium ml-2">{age}</span>
                                )}
                            </h1>
                            {user.gender && (
                                <Badge variant="secondary">{GENDER_LABEL[user.gender] ?? user.gender}</Badge>
                            )}
                        </div>
                        {user.username && (
                            <p className="text-sm text-muted-foreground font-mono">@{user.username}</p>
                        )}
                        <p className="text-xs text-muted-foreground">
                            Member since {new Date(user.createdAt as string).toLocaleDateString("en-US", { month: "long", year: "numeric" })}
                        </p>
                    </div>
                </div>
            }
        >
            <div className="space-y-10">
                {user.interests.length > 0 && (
                    <section className="space-y-3">
                        <h2 className="text-lg font-semibold tracking-tight">Interests</h2>
                        <div className="flex flex-wrap gap-2">
                            {user.interests.map(({ tag }) => {
                                const category = TAG_TO_CATEGORY[tag] ?? "social";
                                const colorClass = CATEGORY_COLORS[category] ?? CATEGORY_COLORS.social;
                                return (
                                    <span
                                        key={tag}
                                        className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${colorClass}`}
                                    >
                                        {tag.replace(/_/g, " ")}
                                    </span>
                                );
                            })}
                        </div>
                    </section>
                )}

                {isOwnProfile && (
                    <>
                        <Separator />
                        <ProfileSettings initialUser={user} />
                    </>
                )}
            </div>
        </Page>
    );
}
