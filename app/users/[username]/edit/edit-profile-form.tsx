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
import { UPDATE_USER, UPDATE_LOCATION } from "@/lib/models/users/gql";
import type { UpdateLocationMutation } from "@/lib/graphql/__generated__/graphql";
import { useHaptics, hapticPatterns } from "@/hooks/useHaptics";
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
import { SUPPORTED_LANGUAGES, type UpdateUserInput } from "@/lib/models/users/validator";
import { MIN_PHOTOS, MIN_PROMPTS } from "@/lib/models/useritems/validator";
import {
    UserItemsEditor,
    userItemsToLocal,
    localPhotosToInput,
    localPromptsToInput,
    type UserItemsState,
} from "./user-items-editor";

export type EditableUser = UpdateUserInput & {
    id: string;
    userItems?: { type: string; content: string; promptKey?: string | null; displayOrder: number }[];
};

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
    const router = useRouter();
    const haptic = useHaptics();

    // Form state
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

    // Items state — initialized from server data, kept local until save
    const initial = userItemsToLocal(user.userItems ?? []);
    const [itemsState, setItemsState] = useState<UserItemsState>({
        photos: initial.photos,
        prompts: initial.prompts,
    });

    // Location state
    const [locatingUser, setLocatingUser] = useState(false);
    const [currentCityName, setCurrentCityName] = useState<string | null>(user.location || null);

    const [updateUser, { loading: saving }] = useMutation(UPDATE_USER);
    const [updateLocation] = useMutation<UpdateLocationMutation>(UPDATE_LOCATION);

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
                    const res = await updateLocation({ variables: { lat: latitude, lon: longitude } });
                    const city = res.data?.updateLocation?.location;
                    if (city) {
                        setCurrentCityName(city);
                        haptic(hapticPatterns.success);
                        toast.success(`Posizione rilevata: ${city}`);
                    } else {
                        haptic(hapticPatterns.success);
                        toast.success("Posizione aggiornata");
                    }
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
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        );
    };

    async function handleSave() {
        if (itemsState.photos.length < MIN_PHOTOS) {
            toast.error(`Aggiungi almeno ${MIN_PHOTOS} foto prima di salvare`);
            return;
        }
        if (itemsState.prompts.length < MIN_PROMPTS) {
            toast.error(`Aggiungi almeno ${MIN_PROMPTS} prompt prima di salvare`);
            return;
        }
        try {
            await updateUser({
                variables: {
                    id: user.id,
                    input: {
                        heightCm: height ? parseInt(height, 10) : undefined,
                        jobTitle: jobTitle.trim() || undefined,
                        schoolName: schoolName.trim() || undefined,
                        sexualOrientation: Array.from(selectedOrientations),
                        relationshipIntent: Array.from(selectedIntents),
                        languages: Array.from(selectedLanguages),
                        photos: localPhotosToInput(itemsState.photos),
                        prompts: localPromptsToInput(itemsState.prompts),
                        ...Object.fromEntries(
                            (Object.keys(enumFields) as EnumKey[])
                                .map((k) => [k, enumFields[k] ?? undefined])
                        ),
                    },
                },
            });
            toast.success("Profilo aggiornato con successo");
            haptic(hapticPatterns.success);
            router.push(`/users/${user.username}`);
            router.refresh();
        } catch (err) {
            console.error("Error updating user", err);
            toast.error("Errore: impossibile salvare il profilo");
        }
    }

    const photoCount = itemsState.photos.length;
    const promptCount = itemsState.prompts.length;

    return (
        <div className="space-y-10">
            {/* ── Foto e Prompts ─────────────────────────────────────────── */}
            <UserItemsEditor
                initialPhotos={itemsState.photos}
                initialPrompts={itemsState.prompts}
                onChange={setItemsState}
            />

            <Separator />

            {/* ── Posizione ────────────────────────────────────────────── */}
            <section className="space-y-4">
                <h3 className="text-xl font-semibold">Posizione attuale</h3>
                <div className="rounded-xl border bg-card p-6 flex flex-col md:flex-row items-center gap-4 justify-between">
                    <div>
                        <p className="font-medium">{currentCityName ?? "Posizione sconosciuta"}</p>
                        <p className="text-sm text-muted-foreground mt-1">Aggiorna le coordinate per scoprire Spazi, Eventi e Persone intorno a te. Utilizziamo OpenStreetMap.</p>
                    </div>
                    <Button onClick={handleUpdateLocation} disabled={locatingUser} className="shrink-0">
                        {locatingUser ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <MapPin className="w-4 h-4 mr-2" />}
                        {locatingUser ? "Rilevamento in corso..." : "Aggiorna Posizione"}
                    </Button>
                </div>
            </section>

            <Separator />

            {/* ── Generalità e Background ───────────────────────────────── */}
            <section className="space-y-6">
                <h3 className="text-xl font-semibold">Generalità e Background</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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

            {/* ── Identità e Intenti ────────────────────────────── */}
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

            {/* ── Azioni Finali ───────────────────────────────────────── */}
            <div className="sticky bottom-6 flex justify-end gap-3 mt-10 p-4 bg-background/80 backdrop-blur-xl border rounded-2xl shadow-lg">
                <Button variant="outline" onClick={() => router.push(`/users/${user.username}`)} disabled={saving}>
                    Annulla
                </Button>
                <Button
                    onClick={handleSave}
                    disabled={saving}
                    size="lg"
                    className="min-w-[150px]"
                    title={photoCount < MIN_PHOTOS || promptCount < MIN_PROMPTS
                        ? `Servono almeno ${MIN_PHOTOS} foto e ${MIN_PROMPTS} prompt`
                        : undefined}
                >
                    {saving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    Salva Modifiche
                    {(photoCount < MIN_PHOTOS || promptCount < MIN_PROMPTS) && (
                        <span className="ml-2 text-xs opacity-70">
                            ({photoCount}/{MIN_PHOTOS} foto · {promptCount}/{MIN_PROMPTS} prompt)
                        </span>
                    )}
                </Button>
            </div>
        </div>
    );
}
