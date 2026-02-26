import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { notFound } from "next/navigation";
import { query } from "@/lib/graphql/apollo-client";
import { GET_USER } from "@/lib/models/users/gql";
import { Page } from "@/components/page";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

import { LogoutButton } from "./logout-button";
import { getTranslations } from "next-intl/server";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import {
    Briefcase, GraduationCap, Languages, Globe, BookOpen,
    Cigarette, Wine, Dumbbell, Baby, Users, Search, Sprout,
    Cake, User, Heart, Ruler, Pencil, MapPin
} from "lucide-react";
import type { GetUserQuery, GetUserQueryVariables } from "@/lib/graphql/__generated__/graphql";
import type { LucideIcon } from "lucide-react";
// ─── Helpers ─────────────────────────────────────────────────────────────────

function getAge(birthdate: string): number {
    const today = new Date();
    const birth = new Date(birthdate);
    let age = today.getFullYear() - birth.getFullYear();
    const m = today.getMonth() - birth.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
    return age;
}

type UserData = NonNullable<GetUserQuery["user"]>;
type UserItemData = UserData["userItems"][number];

// ─── Sub-components ───────────────────────────────────────────────────────────

function PhotoCard({ url, alt }: { url: string; alt: string }) {
    return (
        <div className="relative w-full aspect-[4/5] rounded-2xl overflow-hidden bg-muted shadow-sm">
            <Image
                src={url}
                alt={alt}
                fill
                className="object-cover"
                sizes="(max-width: 640px) 100vw, 480px"
            />
        </div>
    );
}

function PromptCard({ question, answer }: { question: string; answer: string }) {
    return (
        <div className="w-full rounded-2xl border bg-card shadow-sm p-6 space-y-3">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                {question}
            </p>
            <p className="text-xl font-semibold leading-snug">{answer}</p>
        </div>
    );
}

/** Compact icon + label chip for the top row */
function Chip({ icon: Icon, label }: { icon: LucideIcon; label: string }) {
    return (
        <span className="flex items-center gap-1.5 text-sm font-medium">
            <Icon className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
            {label}
        </span>
    );
}

/** Hinge-style row: icon + value, separated from others by a divider */
function InfoRow({ icon: Icon, value }: { icon: LucideIcon; value: string }) {
    return (
        <div className="flex items-center gap-3 py-3 text-sm">
            <Icon className="w-4 h-4 text-muted-foreground shrink-0" />
            <span>{value}</span>
        </div>
    );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default async function UserProfilePage({
    params,
}: {
    params: Promise<{ username: string }>;
}) {
    const { username } = await params;

    const [session, { data }, tEnums, tTags, tPrompts, tProfile] = await Promise.all([
        auth.api.getSession({ headers: await headers() }),
        query<GetUserQuery, GetUserQueryVariables>({
            query: GET_USER,
            variables: { username },
        }),
        getTranslations("enums"),
        getTranslations("tags"),
        getTranslations("prompts"),
        getTranslations("profile"),
    ]);

    const user = data?.user;
    if (!user) notFound();

    const sessionUsername = (session?.user as Record<string, unknown>)?.username as string | undefined;
    const isOwnProfile = !!sessionUsername && sessionUsername === username;

    const age = user.birthdate ? getAge(user.birthdate) : null;
    const initials = (user.name ?? "").split(" ").map((w: string) => w[0]).join("").slice(0, 2).toUpperCase();

    const userItems: UserItemData[] = user.userItems
        .slice()
        .sort((a, b) => a.displayOrder - b.displayOrder);

    // ─── Top row chips (icon + label) ────────────────────────────────────────
    type ChipEntry = { icon: LucideIcon; label: string };
    const chips: ChipEntry[] = [
        ...(age !== null ? [{ icon: Cake, label: String(age) }] : []),
        ...(user.gender ? [{ icon: User, label: tEnums(`gender.${user.gender}` as Parameters<typeof tEnums>[0]) }] : []),
        ...(user.sexualOrientation?.length ? [{ icon: Heart, label: user.sexualOrientation.map((o: string) => tEnums(`sexualOrientation.${o}` as Parameters<typeof tEnums>[0])).join(", ") }] : []),
        ...(user.heightCm ? [{ icon: Ruler, label: `${user.heightCm} cm` }] : []),
        ...(user.hasChildren ? [{ icon: Baby, label: tEnums(`hasChildren.${user.hasChildren}` as Parameters<typeof tEnums>[0]) }] : []),
        ...(user.wantsChildren ? [{ icon: Sprout, label: tEnums(`wantsChildren.${user.wantsChildren}` as Parameters<typeof tEnums>[0]) }] : []),
        ...(user.smoking ? [{ icon: Cigarette, label: tEnums(`smoking.${user.smoking}` as Parameters<typeof tEnums>[0]) }] : []),
        ...(user.drinking ? [{ icon: Wine, label: tEnums(`drinking.${user.drinking}` as Parameters<typeof tEnums>[0]) }] : []),
        ...(user.activityLevel ? [{ icon: Dumbbell, label: tEnums(`activityLevel.${user.activityLevel}` as Parameters<typeof tEnums>[0]) }] : []),
    ];

    // ─── Hinge-style list rows ────────────────────────────────────────────────
    type RowEntry = { icon: LucideIcon; value: string };
    const rows: RowEntry[] = [
        ...(user.location ? [{ icon: MapPin, value: user.location }] : []),
        ...(user.jobTitle ? [{ icon: Briefcase, value: user.jobTitle }] : []),
        ...(user.educationLevel ? [{
            icon: GraduationCap,
            value: [
                tEnums(`educationLevel.${user.educationLevel}` as Parameters<typeof tEnums>[0]),
                user.schoolName,
            ].filter(Boolean).join(" · "),
        }] : user.schoolName ? [{ icon: GraduationCap, value: user.schoolName }] : []),
        ...(user.religion ? [{ icon: BookOpen, value: tEnums(`religion.${user.religion}` as Parameters<typeof tEnums>[0]) }] : []),
        ...(user.languages?.length ? [{ icon: Languages, value: user.languages.map((l: string) => tEnums(`language.${l}` as Parameters<typeof tEnums>[0])).join(", ") }] : []),
        ...(user.ethnicity ? [{ icon: Globe, value: tEnums(`ethnicity.${user.ethnicity}` as Parameters<typeof tEnums>[0]) }] : []),
        ...(user.relationshipIntent?.length ? [{ icon: Search, value: user.relationshipIntent.map((i: string) => tEnums(`relationshipIntent.${i}` as Parameters<typeof tEnums>[0])).join(", ") }] : []),
        ...(user.relationshipStyle ? [{ icon: Users, value: tEnums(`relationshipStyle.${user.relationshipStyle}` as Parameters<typeof tEnums>[0]) }] : []),
    ];



    return (
        <Page breadcrumbs={[{ label: isOwnProfile ? "Il mio profilo" : (user.name ?? "") }]}>
            <div className="pb-16 space-y-6">

                {/* ── Container largo: header + caratteristiche ─────────────── */}
                <div className="space-y-4">

                    {/* ── Hero header ──────────────────────────────────────────── */}
                    <div className="flex items-center gap-4 pb-2">
                        <Avatar className="h-16 w-16 rounded-2xl border-2 border-background shadow-md shrink-0">
                            <AvatarImage src={user.image ?? undefined} alt={user.name ?? ""} />
                            <AvatarFallback className="rounded-2xl text-lg font-bold">{initials}</AvatarFallback>
                        </Avatar>
                        <div className="min-w-0 flex-1">
                            <div className="flex items-baseline gap-2 flex-wrap">
                                <h1 className="text-2xl font-bold tracking-tight truncate">{user.name}</h1>
                            </div>
                            {user.username && (
                                <p className="text-xs text-muted-foreground font-mono mt-0.5">@{user.username}</p>
                            )}
                        </div>
                        {isOwnProfile && (
                            <div className="flex items-center gap-2">
                                <Button variant="outline" size="sm" asChild>
                                    <Link href={`/users/${user.username}/edit`}>
                                        <Pencil className="w-4 h-4 mr-2" />
                                        Modifica Profilo
                                    </Link>
                                </Button>
                                <LogoutButton />
                            </div>
                        )}
                    </div>

                    {/* ── Info card (caratteristiche) ──────────────────────────── */}
                    {(chips.length > 0 || rows.length > 0 || isOwnProfile) && (
                        <div className="space-y-2">
                            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground px-1">
                                {tProfile("sections.aboutMe")}
                            </p>
                            <div className="rounded-2xl border bg-card overflow-hidden">

                                {/* Top row: age · gender · orientation · height */}
                                {chips.length > 0 && (
                                    <div className="flex items-center gap-3 px-4 py-3 border-b flex-wrap">
                                        {chips.map((chip, i) => (
                                            <span key={i} className="flex items-center gap-3">
                                                {i > 0 && <span className="w-px h-4 bg-border shrink-0" />}
                                                <Chip icon={chip.icon} label={chip.label} />
                                            </span>
                                        ))}
                                    </div>
                                )}


                                {/* Hinge-style rows */}
                                {rows.length > 0 && (
                                    <div className="px-4 divide-y">
                                        {rows.map(({ icon, value }, i) => (
                                            <InfoRow key={i} icon={icon} value={value} />
                                        ))}
                                    </div>
                                )}

                            </div>
                        </div>
                    )}

                    {/* ── Tags card ────────────────────────────────────────────── */}
                    {(user.tags?.length ?? 0) > 0 && (
                        <div className="space-y-2">
                            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground px-1">
                                {tProfile("sections.interests")}
                            </p>
                            <div className="rounded-2xl border bg-card px-4 py-3">
                                <div className="flex flex-wrap gap-2">
                                    {user.tags!.map((tag) => (
                                        <span
                                            key={tag}
                                            className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium text-muted-foreground"
                                        >
                                            {tTags(tag as Parameters<typeof tTags>[0])}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                </div>{/* fine container largo */}

                {/* ── Container stretto: foto + prompt ──────────────────────── */}
                <div className="mx-auto max-w-xl space-y-4">

                    {userItems.length === 0 ? (
                        <div className="rounded-2xl border border-dashed p-10 text-center text-muted-foreground text-sm">
                            {tProfile("noContent")}
                        </div>
                    ) : (
                        userItems.map((item) => (
                            <div key={item.id}>
                                {item.type === "photo" ? (
                                    <PhotoCard url={item.content} alt={`Foto di ${user.name}`} />
                                ) : (
                                    <PromptCard
                                        question={item.promptKey ? tPrompts(item.promptKey as Parameters<typeof tPrompts>[0]) : ""}
                                        answer={item.content}
                                    />
                                )}
                            </div>
                        ))
                    )}

                </div>{/* fine container stretto */}

            </div>{/* fine wrapper esterno */}
        </Page>
    );
}
