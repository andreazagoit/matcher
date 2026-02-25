"use client";

import { useState } from "react";
import { useMutation } from "@apollo/client/react";
import { useTranslations } from "next-intl";
import { Loader2, MapPin } from "lucide-react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
    Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { UPDATE_USER, UPDATE_MY_TAGS, UPDATE_LOCATION } from "@/lib/models/users/gql";
import { TAG_CATEGORIES } from "@/lib/models/tags/data";
import { UserItemsEditor } from "./user-items-editor";

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
import type { User } from "@/lib/graphql/__generated__/graphql";

export type EditableUser = Pick<
    User,
    | "id" | "name" | "birthdate" | "username"
    | "gender" | "sexualOrientation" | "heightCm"
    | "relationshipIntent" | "relationshipStyle"
    | "hasChildren" | "wantsChildren"
    | "religion" | "smoking" | "drinking" | "activityLevel"
    | "jobTitle" | "educationLevel" | "schoolName" | "languages" | "ethnicity"
    | "locationText" | "tags"
>;

type Props = {
    user: EditableUser;
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div className="space-y-1.5">
            <Label className="text-sm font-medium">{label}</Label>
            {children}
        </div>
    );
}

type EnumKey =
    | "gender" | "relationshipStyle"
    | "hasChildren" | "wantsChildren" | "religion" | "smoking" | "drinking" | "activityLevel"
    | "educationLevel" | "ethnicity";

const ENUM_VALUES: Record<EnumKey, readonly string[]> = {
    gender: genderEnum.enumValues,
    relationshipStyle: relationshipStyleEnum.enumValues,
    hasChildren: hasChildrenEnum.enumValues,
    wantsChildren: wantsChildrenEnum.enumValues,
    religion: religionEnum.enumValues,
    smoking: smokingEnum.enumValues,
    drinking: drinkingEnum.enumValues,
    activityLevel: activityLevelEnum.enumValues,
    educationLevel: educationLevelEnum.enumValues,
    ethnicity: ethnicityEnum.enumValues,
};

// ─── Component ───────────────────────────────────────────────────────────────

export function EditProfileForm({ user }: Props) {
    const tEnums = useTranslations("enums");
    const tTags = useTranslations("tags");
    const tTagCats = useTranslations("tagCategories");
    const router = useRouter();

    // Form state
    const [name, setName] = useState(user.name ?? "");
    const [height, setHeight] = useState(user.heightCm?.toString() ?? "");
    const [jobTitle, setJobTitle] = useState(user.jobTitle ?? "");
    const [schoolName, setSchoolName] = useState(user.schoolName ?? "");
    const [selectedLanguages, setSelectedLanguages] = useState<Set<string>>(
        new Set(user.languages ?? [])
    );
    const [selectedOrientations, setSelectedOrientations] = useState<Set<string>>(
        new Set(user.sexualOrientation ?? [])
    );
    const [selectedIntents, setSelectedIntents] = useState<Set<string>>(
        new Set(user.relationshipIntent ?? [])
    );
    const [selectedTags, setSelectedTags] = useState<Set<string>>(
        new Set(user.tags ?? [])
    );
    const [enumFields, setEnumFields] = useState<Record<EnumKey, string | null>>({
        gender: user.gender ?? null,
        relationshipStyle: user.relationshipStyle ?? null,
        hasChildren: user.hasChildren ?? null,
        wantsChildren: user.wantsChildren ?? null,
        religion: user.religion ?? null,
        smoking: user.smoking ?? null,
        drinking: user.drinking ?? null,
        activityLevel: user.activityLevel ?? null,
        educationLevel: user.educationLevel ?? null,
        ethnicity: user.ethnicity ?? null,
    });

    // Location state
    const [locatingUser, setLocatingUser] = useState(false);
    const [currentCityName, setCurrentCityName] = useState<string | null>(user.locationText || null);

    const [updateUser, { loading: saving }] = useMutation(UPDATE_USER);
    const [updateMyTags, { loading: savingTags }] = useMutation(UPDATE_MY_TAGS);
    const [updateLocation] = useMutation(UPDATE_LOCATION);

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

    const handleUpdateLocation = () => {
        if (!navigator.geolocation) {
            toast.error("Geolocalizzazione non supportata dal tuo browser.");
            return;
        }

        setLocatingUser(true);
        navigator.geolocation.getCurrentPosition(
            async (position) => {
                const { latitude, longitude } = position.coords;
                try {
                    // Reverse geocoding base call with OpenStreetMap
                    const res = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`);
                    let city: string | null = null;
                    if (res.ok) {
                        const data = await res.json();
                        city = data.address?.city || data.address?.town || data.address?.village || data.address?.county || "Località sconosciuta";
                        setCurrentCityName(city);
                        toast.success(`Posizione rilevata: ${city}`);
                    }

                    // Save to backend
                    await updateLocation({ variables: { lat: latitude, lon: longitude, locationText: city } });

                } catch (err) {
                    console.error("Location update failed", err);
                    toast.error("Errore nell'aggiornamento della posizione");
                } finally {
                    setLocatingUser(false);
                }
            },
            (err) => {
                console.error("GPS error", err);
                toast.error("Impossibile accedere alla tua posizione. Controlla i permessi.");
                setLocatingUser(false);
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );
    };

    async function handleSave() {
        try {
            await Promise.all([
                updateUser({
                    variables: {
                        id: user.id,
                        input: {
                            name: name.trim() || undefined,
                            heightCm: height ? parseInt(height, 10) : undefined,
                            jobTitle: jobTitle.trim() || undefined,
                            schoolName: schoolName.trim() || undefined,
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
                updateMyTags({ variables: { tags: Array.from(selectedTags) } }),
            ]);
            toast.success("Profilo aggiornato con successo");
            router.push(`/users/${user.username}`);
            router.refresh();
        } catch (err) {
            console.error("Error updating user", err);
            toast.error("Errore: impossibile salvare il profilo");
        }
    }

    return (
        <div className="space-y-10">
            {/* ── Posizione ────────────────────────────────────────────── */}
            <section className="space-y-4">
                <h3 className="text-xl font-semibold">Posizione attuale</h3>
                <div className="rounded-xl border bg-card p-6 flex flex-col md:flex-row items-center gap-4 justify-between">
                    <div>
                        <p className="font-medium">{currentCityName ? currentCityName : "Posizione sconosciuta"}</p>
                        <p className="text-sm text-muted-foreground mt-1">Aggiorna le coordinate per scoprire Spazi, Eventi e Persone intorno a te. Utilizziamo OpenStreetMap.</p>
                    </div>
                    <Button onClick={handleUpdateLocation} disabled={locatingUser} className="shrink-0">
                        {locatingUser ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <MapPin className="w-4 h-4 mr-2" />}
                        {locatingUser ? "Rilevamento in corso..." : "Aggiorna Posizione"}
                    </Button>
                </div>
            </section>

            <Separator />

            {/* ── Caratteristiche Generali ───────────────────────────────── */}
            <section className="space-y-6">
                <h3 className="text-xl font-semibold">Generalità e Background</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Field label="Nome">
                        <Input
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="Il tuo nome"
                        />
                    </Field>

                    <Field label="Professione">
                        <Input
                            value={jobTitle}
                            onChange={(e) => setJobTitle(e.target.value)}
                            placeholder="es. Designer"
                        />
                    </Field>

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

                    <Field label="Scuola / Università">
                        <Input
                            value={schoolName}
                            onChange={(e) => setSchoolName(e.target.value)}
                            placeholder="es. Politecnico di Milano"
                        />
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

            {/* ── Identità e Intenti (Multi) ────────────────────────────── */}
            <section className="space-y-6">
                <h3 className="text-xl font-semibold">Identità e Cosa cerchi</h3>
                <div className="space-y-6">
                    <Field label={tEnums("sexualOrientationLabel" as Parameters<typeof tEnums>[0])}>
                        <div className="flex flex-wrap gap-2">
                            {sexualOrientationEnum.enumValues.map((v) => {
                                const active = selectedOrientations.has(v);
                                return (
                                    <button key={v} type="button" onClick={() => toggleOrientation(v)}
                                        className={["inline-flex items-center rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                                            active ? "bg-primary text-primary-foreground border-primary" : "text-muted-foreground hover:bg-accent/50"].join(" ")}>
                                        {tEnums(`sexualOrientation.${v}` as Parameters<typeof tEnums>[0])}
                                    </button>
                                );
                            })}
                        </div>
                    </Field>

                    <Field label={tEnums("relationshipIntentLabel" as Parameters<typeof tEnums>[0])}>
                        <div className="flex flex-wrap gap-2">
                            {relationshipIntentEnum.enumValues.map((v) => {
                                const active = selectedIntents.has(v);
                                return (
                                    <button key={v} type="button" onClick={() => toggleIntent(v)}
                                        className={["inline-flex items-center rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                                            active ? "bg-primary text-primary-foreground border-primary" : "text-muted-foreground hover:bg-accent/50"].join(" ")}>
                                        {tEnums(`relationshipIntent.${v}` as Parameters<typeof tEnums>[0])}
                                    </button>
                                );
                            })}
                        </div>
                    </Field>
                </div>
            </section>

            <Separator />

            {/* ── Interessi / Tags ──────────────────────────────────────── */}
            <section className="space-y-6">
                <div className="flex items-center justify-between">
                    <h3 className="text-xl font-semibold">Interessi e Hobby</h3>
                    {selectedTags.size > 0 && (
                        <span className="text-sm font-medium text-primary bg-primary/10 px-3 py-1 rounded-full">
                            {selectedTags.size} selezionati
                        </span>
                    )}
                </div>
                <div className="space-y-6">
                    {Object.entries(TAG_CATEGORIES).map(([category, tags]) => (
                        <div key={category} className="space-y-3">
                            <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
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
                                                "inline-flex items-center rounded-full border px-3 py-1.5 text-sm font-medium transition-colors",
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

            <Separator />

            {/* ── Lingue ─────────────────────────────────────────── */}
            <section className="space-y-4">
                <div className="flex items-center justify-between">
                    <h3 className="text-xl font-semibold">Lingue parlate</h3>
                    {selectedLanguages.size > 0 && (
                        <span className="text-sm font-medium text-primary bg-primary/10 px-3 py-1 rounded-full">
                            {selectedLanguages.size} selezionate
                        </span>
                    )}
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                    {SUPPORTED_LANGUAGES.map((lang) => {
                        const active = selectedLanguages.has(lang);
                        return (
                            <button
                                key={lang}
                                type="button"
                                onClick={() => toggleLanguage(lang)}
                                className={[
                                    "inline-flex items-center rounded-full border px-4 py-2 text-sm font-medium transition-colors",
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

            {/* ── Foto e Prompts ─────────────────────────────────────────── */}
            <UserItemsEditor userId={user.id} />

            {/* ── Azioni Finali ───────────────────────────────────────── */}
            <div className="sticky bottom-6 flex justify-end gap-3 mt-10 p-4 bg-background/80 backdrop-blur-xl border rounded-2xl shadow-lg">
                <Button variant="outline" onClick={() => router.push(`/users/${user.username}`)} disabled={saving || savingTags}>
                    Annulla
                </Button>
                <Button onClick={handleSave} disabled={saving || savingTags} size="lg" className="min-w-[150px]">
                    {(saving || savingTags) && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    Salva Modifiche
                </Button>
            </div>
        </div>
    );
}
