"use client";

import { useState } from "react";
import { useMutation } from "@apollo/client/react";
import { useTranslations } from "next-intl";
import { Pencil, Loader2 } from "lucide-react";

import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger, SheetFooter } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
    Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

import { UPDATE_USER } from "@/lib/models/users/gql";
import { UPDATE_MY_INTERESTS } from "@/lib/models/interests/gql";
import { TAG_CATEGORIES } from "@/lib/models/tags/data";
import {
    genderEnum,
    sexualOrientationEnum,
    relationshipIntentEnum,
    relationshipStyleEnum,
    hasChildrenEnum,
    wantsChildrenEnum,
    religionEnum,
    smokingEnum,
    drinkingEnum,
    activityLevelEnum,
    educationLevelEnum,
    ethnicityEnum,
} from "@/lib/models/users/schema";
import { SUPPORTED_LANGUAGES } from "@/lib/models/users/validator";
import type { User, UserInterest } from "@/lib/graphql/__generated__/graphql";

// ─── Types ───────────────────────────────────────────────────────────────────

export type EditableUser = Pick<
    User,
    | "id" | "name" | "birthdate" | "username"
    | "gender" | "sexualOrientation" | "heightCm"
    | "relationshipIntent" | "relationshipStyle"
    | "hasChildren" | "wantsChildren"
    | "religion" | "smoking" | "drinking" | "activityLevel"
    | "jobTitle" | "educationLevel" | "languages" | "ethnicity"
>;

type Props = {
    user: EditableUser;
    interests: UserInterest[];
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">{label}</Label>
            {children}
        </div>
    );
}

// Single-select fields (stored as single value in DB)
type EnumKey =
    | "gender" | "relationshipStyle"
    | "hasChildren" | "wantsChildren" | "religion" | "smoking" | "drinking" | "activityLevel"
    | "educationLevel" | "ethnicity";

const ENUM_VALUES: Record<EnumKey, readonly string[]> = {
    gender:             genderEnum.enumValues,
    relationshipStyle:  relationshipStyleEnum.enumValues,
    hasChildren:        hasChildrenEnum.enumValues,
    wantsChildren:      wantsChildrenEnum.enumValues,
    religion:           religionEnum.enumValues,
    smoking:            smokingEnum.enumValues,
    drinking:           drinkingEnum.enumValues,
    activityLevel:      activityLevelEnum.enumValues,
    educationLevel:     educationLevelEnum.enumValues,
    ethnicity:          ethnicityEnum.enumValues,
};

// ─── Component ───────────────────────────────────────────────────────────────

export function EditProfileSheet({ user, interests }: Props) {
    const tEnums = useTranslations("enums");
    const tTags = useTranslations("tags");
    const tTagCats = useTranslations("tagCategories");

    const [open, setOpen] = useState(false);

    // Form state
    const [name, setName] = useState(user.name ?? "");
    const [birthdate, setBirthdate] = useState(user.birthdate ?? "");
    const [height, setHeight] = useState(user.heightCm?.toString() ?? "");
    const [jobTitle, setJobTitle] = useState(user.jobTitle ?? "");
    const [selectedLanguages, setSelectedLanguages] = useState<Set<string>>(
        new Set(user.languages ?? [])
    );
    const [selectedOrientations, setSelectedOrientations] = useState<Set<string>>(
        new Set(user.sexualOrientation ?? [])
    );
    const [selectedIntents, setSelectedIntents] = useState<Set<string>>(
        new Set(user.relationshipIntent ?? [])
    );
    const [enumFields, setEnumFields] = useState<Record<EnumKey, string | null>>({
        gender:             user.gender ?? null,
        relationshipStyle:  user.relationshipStyle ?? null,
        hasChildren:        user.hasChildren ?? null,
        wantsChildren:      user.wantsChildren ?? null,
        religion:           user.religion ?? null,
        smoking:            user.smoking ?? null,
        drinking:           user.drinking ?? null,
        activityLevel:      user.activityLevel ?? null,
        educationLevel:     user.educationLevel ?? null,
        ethnicity:          user.ethnicity ?? null,
    });
    const [selectedTags, setSelectedTags] = useState<Set<string>>(
        new Set(interests.map((i) => i.tag))
    );

    const [updateUser, { loading: savingUser }] = useMutation(UPDATE_USER);
    const [updateInterests, { loading: savingInterests }] = useMutation(UPDATE_MY_INTERESTS);
    const saving = savingUser || savingInterests;

    function toggleSet(setter: React.Dispatch<React.SetStateAction<Set<string>>>, value: string) {
        setter((prev) => {
            const next = new Set(prev);
            if (next.has(value)) next.delete(value);
            else next.add(value);
            return next;
        });
    }

    function toggleLanguage(lang: string) { toggleSet(setSelectedLanguages, lang); }
    function toggleOrientation(v: string) { toggleSet(setSelectedOrientations, v); }
    function toggleIntent(v: string) { toggleSet(setSelectedIntents, v); }
    function toggleTag(tag: string) { toggleSet(setSelectedTags, tag); }

    function setEnum(key: EnumKey, value: string) {
        setEnumFields((prev) => ({ ...prev, [key]: value === "__clear__" ? null : value }));
    }

    async function handleSave() {
        await Promise.all([
            updateUser({
                variables: {
                    id: user.id,
                    input: {
                        name: name.trim() || undefined,
                        birthdate: birthdate || undefined,
                        heightCm: height ? parseInt(height, 10) : undefined,
                        jobTitle: jobTitle.trim() || undefined,
                        sexualOrientation: Array.from(selectedOrientations),
                        relationshipIntent: Array.from(selectedIntents),
                        languages: Array.from(selectedLanguages),
                        ...Object.fromEntries(
                            (Object.keys(enumFields) as EnumKey[])
                                .map((k) => [k, enumFields[k] ?? undefined])
                        ),
                    },
                },
            }),
            updateInterests({
                variables: { tags: Array.from(selectedTags) },
            }),
        ]);
        setOpen(false);
        // Reload to reflect server changes
        window.location.reload();
    }

    return (
        <Sheet open={open} onOpenChange={setOpen}>
            <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="h-7 w-7">
                    <Pencil className="w-3.5 h-3.5" />
                </Button>
            </SheetTrigger>

            <SheetContent className="w-full sm:max-w-md overflow-y-auto flex flex-col gap-0 p-0">
                <SheetHeader className="px-6 py-5 border-b">
                    <SheetTitle>Modifica profilo</SheetTitle>
                </SheetHeader>

                <div className="flex-1 overflow-y-auto px-6 py-5 space-y-7">

                    {/* ── Info base ──────────────────────────────────────── */}
                    <section className="space-y-4">
                        <h3 className="text-sm font-semibold">Informazioni</h3>
                        <div className="grid grid-cols-1 gap-4">
                            <Field label="Nome">
                                <Input
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    placeholder="Il tuo nome"
                                />
                            </Field>
                            <Field label="Data di nascita">
                                <Input
                                    type="date"
                                    value={birthdate}
                                    onChange={(e) => setBirthdate(e.target.value)}
                                />
                            </Field>
                            <Field label="Professione">
                                <Input
                                    value={jobTitle}
                                    onChange={(e) => setJobTitle(e.target.value)}
                                    placeholder="es. Ingegnere software"
                                />
                            </Field>
                        </div>
                    </section>

                    <Separator />

                    {/* ── Caratteristiche ────────────────────────────────── */}
                    <section className="space-y-4">
                        <h3 className="text-sm font-semibold">Caratteristiche</h3>
                        <div className="grid grid-cols-1 gap-4">

                            {/* Height (numeric) */}
                            <Field label="Altezza (cm)">
                                <Input
                                    type="number"
                                    min={100}
                                    max={250}
                                    value={height}
                                    onChange={(e) => setHeight(e.target.value)}
                                    placeholder="es. 175"
                                />
                            </Field>

                            {/* Orientamento sessuale (multi) */}
                            <Field label={tEnums("sexualOrientationLabel" as Parameters<typeof tEnums>[0])}>
                                <div className="flex flex-wrap gap-2">
                                    {sexualOrientationEnum.enumValues.map((v) => {
                                        const active = selectedOrientations.has(v);
                                        return (
                                            <button key={v} type="button" onClick={() => toggleOrientation(v)}
                                                className={["inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                                                    active ? "bg-foreground text-background border-foreground" : "text-muted-foreground hover:border-foreground/40"].join(" ")}>
                                                {tEnums(`sexualOrientation.${v}` as Parameters<typeof tEnums>[0])}
                                            </button>
                                        );
                                    })}
                                </div>
                            </Field>

                            {/* Cosa cerco (multi) */}
                            <Field label={tEnums("relationshipIntentLabel" as Parameters<typeof tEnums>[0])}>
                                <div className="flex flex-wrap gap-2">
                                    {relationshipIntentEnum.enumValues.map((v) => {
                                        const active = selectedIntents.has(v);
                                        return (
                                            <button key={v} type="button" onClick={() => toggleIntent(v)}
                                                className={["inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                                                    active ? "bg-foreground text-background border-foreground" : "text-muted-foreground hover:border-foreground/40"].join(" ")}>
                                                {tEnums(`relationshipIntent.${v}` as Parameters<typeof tEnums>[0])}
                                            </button>
                                        );
                                    })}
                                </div>
                            </Field>

                            {/* All single-select enums */}
                            {(Object.keys(ENUM_VALUES) as EnumKey[]).map((key) => (
                                <Field key={key} label={tEnums(`${key}Label` as Parameters<typeof tEnums>[0])}>
                                    <Select
                                        value={enumFields[key] ?? ""}
                                        onValueChange={(v) => setEnum(key, v)}
                                    >
                                        <SelectTrigger>
                                            <SelectValue placeholder="—" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="__clear__">
                                                <span className="text-muted-foreground">— Non specificato</span>
                                            </SelectItem>
                                            {ENUM_VALUES[key].map((v) => (
                                                <SelectItem key={v} value={v}>
                                                    {tEnums(`${key}.${v}` as Parameters<typeof tEnums>[0])}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </Field>
                            ))}

                        </div>
                    </section>

                    <Separator />

                    {/* ── Lingue ─────────────────────────────────────────── */}
                    <section className="space-y-3">
                        <h3 className="text-sm font-semibold">
                            Lingue parlate
                            <span className="ml-2 text-xs font-normal text-muted-foreground">
                                {selectedLanguages.size > 0 ? `${selectedLanguages.size} selezionate` : ""}
                            </span>
                        </h3>
                        <div className="flex flex-wrap gap-2">
                            {SUPPORTED_LANGUAGES.map((lang) => {
                                const active = selectedLanguages.has(lang);
                                return (
                                    <button
                                        key={lang}
                                        type="button"
                                        onClick={() => toggleLanguage(lang)}
                                        className={[
                                            "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                                            active
                                                ? "bg-foreground text-background border-foreground"
                                                : "text-muted-foreground hover:border-foreground/40",
                                        ].join(" ")}
                                    >
                                        {tEnums(`language.${lang}` as Parameters<typeof tEnums>[0])}
                                    </button>
                                );
                            })}
                        </div>
                    </section>

                    <Separator />

                    {/* ── Interessi ──────────────────────────────────────── */}
                    <section className="space-y-4">
                        <h3 className="text-sm font-semibold">
                            Interessi
                            <span className="ml-2 text-xs font-normal text-muted-foreground">
                                {selectedTags.size} selezionati
                            </span>
                        </h3>
                        <div className="space-y-4">
                            {Object.entries(TAG_CATEGORIES).map(([category, tags]) => (
                                <div key={category} className="space-y-2">
                                    <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
                                        {tTagCats(category as Parameters<typeof tTagCats>[0])}
                                    </p>
                                    <div className="flex flex-wrap gap-2">
                                        {tags.map((tag) => {
                                            const active = selectedTags.has(tag);
                                            return (
                                                <button
                                                    key={tag}
                                                    type="button"
                                                    onClick={() => toggleTag(tag)}
                                                    className={[
                                                        "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                                                        active
                                                            ? "bg-foreground text-background border-foreground"
                                                            : "text-muted-foreground hover:border-foreground/40",
                                                    ].join(" ")}
                                                >
                                                    {tTags(tag as Parameters<typeof tTags>[0])}
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>

                </div>

                <SheetFooter className="px-6 py-4 border-t">
                    <Button variant="outline" onClick={() => setOpen(false)} disabled={saving}>
                        Annulla
                    </Button>
                    <Button onClick={handleSave} disabled={saving}>
                        {saving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                        Salva
                    </Button>
                </SheetFooter>
            </SheetContent>
        </Sheet>
    );
}
