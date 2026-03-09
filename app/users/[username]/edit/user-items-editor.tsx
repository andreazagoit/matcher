"use client";

import { useState, useCallback } from "react";
import {
    DndContext,
    closestCenter,
    MouseSensor,
    TouchSensor,
    useSensor,
    useSensors,
    DragOverlay,
    type DragStartEvent,
    type DragEndEvent,
} from "@dnd-kit/core";
import {
    SortableContext,
    useSortable,
    rectSortingStrategy,
    verticalListSortingStrategy,
    arrayMove,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
    Trash2, Image as ImageIcon, MessageSquare,
    AlertCircle, GripVertical, Plus,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
    Select, SelectContent, SelectGroup, SelectItem,
    SelectLabel, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { useTranslations } from "next-intl";
import { useHaptics, hapticPatterns } from "@/hooks/useHaptics";

import { PROMPTS_BY_CATEGORY, CATEGORY_LABELS } from "@/lib/models/useritems/prompts";
import { MAX_PHOTOS, MAX_PROMPTS, MIN_PHOTOS, MIN_PROMPTS } from "@/lib/models/useritems/validator";
import type { PhotoItemInput, PromptItemInput } from "@/lib/models/useritems/validator";

// ─── Types ────────────────────────────────────────────────────────────────────

// Local-only item — no id/userId/displayOrder (those are DB concerns)
export interface LocalPhoto {
    // Temporary client-only key for DnD identity (not persisted)
    _key: string;
    content: string;
}

export interface LocalPrompt {
    _key: string;
    promptKey: string;
    content: string;
}

export interface UserItemsState {
    photos: LocalPhoto[];
    prompts: LocalPrompt[];
}

interface Props {
    initialPhotos: LocalPhoto[];
    initialPrompts: LocalPrompt[];
    onChange: (state: UserItemsState) => void;
}

let _keyCounter = 0;
export function makeKey() { return `local-${++_keyCounter}`; }

// ─── Sortable photo tile ──────────────────────────────────────────────────────

function SortablePhoto({
    photo,
    index,
    onDelete,
    isDragOverlay = false,
}: {
    photo: LocalPhoto;
    index: number;
    onDelete: (key: string) => void;
    isDragOverlay?: boolean;
}) {
    const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
        useSortable({ id: photo._key });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition: isDragOverlay ? undefined : transition,
    };

    return (
        <div
            ref={isDragOverlay ? undefined : setNodeRef}
            style={style}
            className={[
                "group relative aspect-[4/5] rounded-xl overflow-hidden border bg-muted select-none",
                isDragging && !isDragOverlay ? "opacity-30" : "",
                isDragOverlay ? "shadow-2xl rotate-1 cursor-grabbing" : "",
            ].join(" ")}
            {...(isDragOverlay ? {} : attributes)}
            {...(isDragOverlay ? {} : listeners)}
        >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
                src={photo.content}
                alt={`Foto ${index + 1}`}
                className="w-full h-full object-cover pointer-events-none"
                draggable={false}
            />
            {!isDragOverlay && (
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors" />
            )}
            <div className="absolute top-2 left-2 w-6 h-6 rounded-full bg-black/50 text-white text-xs font-bold flex items-center justify-center pointer-events-none">
                {index + 1}
            </div>
            {!isDragOverlay && (
                <div className="absolute top-2 right-2 w-7 h-7 rounded bg-black/40 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                    <GripVertical className="w-4 h-4" />
                </div>
            )}
            {!isDragOverlay && (
                <button
                    type="button"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => { e.stopPropagation(); onDelete(photo._key); }}
                    className="absolute bottom-2 right-2 w-7 h-7 rounded-full bg-black/60 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive"
                    aria-label="Rimuovi foto"
                >
                    <Trash2 className="w-3.5 h-3.5" />
                </button>
            )}
        </div>
    );
}

// ─── Sortable prompt row ──────────────────────────────────────────────────────

function SortablePrompt({
    prompt,
    onDelete,
    isDragOverlay = false,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tPrompts,
}: {
    prompt: LocalPrompt;
    onDelete: (key: string) => void;
    isDragOverlay?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tPrompts: (key: any) => string;
}) {
    const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
        useSortable({ id: prompt._key });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition: isDragOverlay ? undefined : transition,
    };

    return (
        <div
            ref={isDragOverlay ? undefined : setNodeRef}
            style={style}
            className={[
                "group flex gap-3 rounded-xl border bg-card p-4 select-none",
                isDragging && !isDragOverlay ? "opacity-30" : "",
                isDragOverlay ? "shadow-2xl -rotate-1 cursor-grabbing" : "",
            ].join(" ")}
        >
            <button
                type="button"
                {...(isDragOverlay ? {} : attributes)}
                {...(isDragOverlay ? {} : listeners)}
                onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)}
                className="text-muted-foreground opacity-40 group-hover:opacity-100 transition-opacity cursor-grab active:cursor-grabbing mt-0.5 shrink-0 touch-none"
                aria-label="Trascina per riordinare"
            >
                <GripVertical className="w-4 h-4" />
            </button>
            <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-muted-foreground mb-1.5 flex items-center gap-1.5">
                    <MessageSquare className="w-3 h-3" />
                    {tPrompts(prompt.promptKey)}
                </p>
                <p className="text-sm leading-relaxed">{prompt.content}</p>
            </div>
            {!isDragOverlay && (
                <button
                    type="button"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => { e.stopPropagation(); onDelete(prompt._key); }}
                    className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive mt-0.5 shrink-0"
                    aria-label="Rimuovi prompt"
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            )}
        </div>
    );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function UserItemsEditor({ initialPhotos, initialPrompts, onChange }: Props) {
    const tPrompts = useTranslations("prompts");
    const haptic = useHaptics();

    const [photos, setPhotos] = useState<LocalPhoto[]>(initialPhotos);
    const [prompts, setPrompts] = useState<LocalPrompt[]>(initialPrompts);
    const [activePhoto, setActivePhoto] = useState<LocalPhoto | null>(null);
    const [activePrompt, setActivePrompt] = useState<LocalPrompt | null>(null);

    // Inline add photo
    const [showAddPhoto, setShowAddPhoto] = useState(false);
    const [newPhotoUrl, setNewPhotoUrl] = useState("");

    // Inline add prompt
    const [showAddPrompt, setShowAddPrompt] = useState(false);
    const [newPromptKey, setNewPromptKey] = useState("");
    const [newPromptContent, setNewPromptContent] = useState("");

    const notify = useCallback((nextPhotos: LocalPhoto[], nextPrompts: LocalPrompt[]) => {
        onChange({ photos: nextPhotos, prompts: nextPrompts });
    }, [onChange]);

    const sensors = useSensors(
        useSensor(MouseSensor, { activationConstraint: { distance: 8 } }),
        useSensor(TouchSensor, { activationConstraint: { delay: 200, tolerance: 5 } }),
    );

    // ── Photo DnD ─────────────────────────────────────────────────────────────

    const handlePhotoDragStart = useCallback((event: DragStartEvent) => {
        setActivePhoto(photos.find((p) => p._key === event.active.id) ?? null);
    }, [photos]);

    const handlePhotoDragEnd = useCallback((event: DragEndEvent) => {
        setActivePhoto(null);
        const { active, over } = event;
        if (!over || active.id === over.id) return;
        const oldIndex = photos.findIndex((p) => p._key === active.id);
        const newIndex = photos.findIndex((p) => p._key === over.id);
        const next = arrayMove(photos, oldIndex, newIndex);
        haptic(hapticPatterns.drop);
        setPhotos(next);
        notify(next, prompts);
    }, [photos, prompts, notify, haptic]);

    // ── Prompt DnD ────────────────────────────────────────────────────────────

    const handlePromptDragStart = useCallback((event: DragStartEvent) => {
        setActivePrompt(prompts.find((p) => p._key === event.active.id) ?? null);
    }, [prompts]);

    const handlePromptDragEnd = useCallback((event: DragEndEvent) => {
        setActivePrompt(null);
        const { active, over } = event;
        if (!over || active.id === over.id) return;
        const oldIndex = prompts.findIndex((p) => p._key === active.id);
        const newIndex = prompts.findIndex((p) => p._key === over.id);
        const next = arrayMove(prompts, oldIndex, newIndex);
        haptic(hapticPatterns.drop);
        setPrompts(next);
        notify(photos, next);
    }, [photos, prompts, notify, haptic]);

    // ── Add photo ─────────────────────────────────────────────────────────────

    const handleAddPhoto = () => {
        const url = newPhotoUrl.trim();
        if (!url) return;
        const next = [...photos, { _key: makeKey(), content: url }];
        haptic(hapticPatterns.confirm);
        setPhotos(next);
        notify(next, prompts);
        setNewPhotoUrl("");
        setShowAddPhoto(false);
    };

    const handleAddPrompt = () => {
        if (!newPromptKey || !newPromptContent.trim()) return;
        const next = [...prompts, { _key: makeKey(), promptKey: newPromptKey, content: newPromptContent.trim() }];
        haptic(hapticPatterns.confirm);
        setPrompts(next);
        notify(photos, next);
        setNewPromptKey("");
        setNewPromptContent("");
        setShowAddPrompt(false);
    };

    const handleDeletePhoto = (key: string) => {
        haptic(hapticPatterns.delete);
        const next = photos.filter((p) => p._key !== key);
        setPhotos(next);
        notify(next, prompts);
    };

    const handleDeletePrompt = (key: string) => {
        haptic(hapticPatterns.delete);
        const next = prompts.filter((p) => p._key !== key);
        setPrompts(next);
        notify(photos, next);
    };

    const usedPromptKeys = new Set(prompts.map((p) => p.promptKey));

    return (
        <div className="space-y-8">
            {/* ── Validation banner ── */}
            {(photos.length < MIN_PHOTOS || prompts.length < MIN_PROMPTS) && (
                <div className="flex gap-3 items-start rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-amber-600 dark:text-amber-400">
                    <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                    <span>
                        Per pubblicare il profilo servono almeno{" "}
                        <strong>{MIN_PHOTOS} foto</strong> e <strong>{MIN_PROMPTS} prompt</strong>.{" "}
                        {photos.length < MIN_PHOTOS && `Mancano ${MIN_PHOTOS - photos.length} foto. `}
                        {prompts.length < MIN_PROMPTS && `Mancano ${MIN_PROMPTS - prompts.length} prompt.`}
                    </span>
                </div>
            )}

            {/* ── Foto ── */}
            <section className="space-y-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <h3 className="text-xl font-semibold">Foto</h3>
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                            photos.length < MIN_PHOTOS
                                ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                                : "bg-muted text-muted-foreground"
                        }`}>
                            {photos.length} / {MAX_PHOTOS}
                        </span>
                    </div>
                    {photos.length < MAX_PHOTOS && !showAddPhoto && (
                        <Button variant="outline" size="sm" onClick={() => setShowAddPhoto(true)}>
                            <Plus className="w-3.5 h-3.5 mr-1.5" />
                            Aggiungi foto
                        </Button>
                    )}
                </div>

                {showAddPhoto && (
                    <div className="rounded-xl border bg-card p-4 space-y-3">
                        <div className="space-y-1.5">
                            <label className="text-sm font-medium">URL immagine</label>
                            <Input
                                placeholder="https://…"
                                value={newPhotoUrl}
                                onChange={(e) => setNewPhotoUrl(e.target.value)}
                                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); handleAddPhoto(); } }}
                                autoFocus
                            />
                        </div>
                        <div className="flex gap-2 justify-end">
                            <Button variant="ghost" size="sm" onClick={() => { setShowAddPhoto(false); setNewPhotoUrl(""); }}>
                                Annulla
                            </Button>
                            <Button size="sm" onClick={handleAddPhoto} disabled={!newPhotoUrl.trim()}>
                                Aggiungi
                            </Button>
                        </div>
                    </div>
                )}

                {photos.length === 0 ? (
                    <button
                        type="button"
                        onClick={() => setShowAddPhoto(true)}
                        className="w-full rounded-xl border border-dashed py-10 text-center text-sm text-muted-foreground hover:border-foreground/30 hover:text-foreground transition-colors"
                    >
                        <ImageIcon className="w-6 h-6 mx-auto mb-2 opacity-40" />
                        Nessuna foto — aggiungine almeno {MIN_PHOTOS}
                    </button>
                ) : (
                    <DndContext
                        sensors={sensors}
                        collisionDetection={closestCenter}
                        onDragStart={handlePhotoDragStart}
                        onDragEnd={handlePhotoDragEnd}
                    >
                        <SortableContext items={photos.map((p) => p._key)} strategy={rectSortingStrategy}>
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                                {photos.map((photo, idx) => (
                                    <SortablePhoto
                                        key={photo._key}
                                        photo={photo}
                                        index={idx}
                                        onDelete={handleDeletePhoto}
                                    />
                                ))}
                                {photos.length < MAX_PHOTOS && !showAddPhoto && (
                                    <button
                                        type="button"
                                        onClick={() => setShowAddPhoto(true)}
                                        className="aspect-[4/5] rounded-xl border border-dashed flex flex-col items-center justify-center gap-2 text-muted-foreground hover:border-foreground/30 hover:text-foreground transition-colors"
                                    >
                                        <Plus className="w-5 h-5 opacity-40" />
                                        <span className="text-xs">Aggiungi</span>
                                    </button>
                                )}
                            </div>
                        </SortableContext>
                        <DragOverlay>
                            {activePhoto && (
                                <SortablePhoto
                                    photo={activePhoto}
                                    index={photos.indexOf(activePhoto)}
                                    onDelete={() => {}}
                                    isDragOverlay
                                />
                            )}
                        </DragOverlay>
                    </DndContext>
                )}
            </section>

            {/* ── Prompt ── */}
            <section className="space-y-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <h3 className="text-xl font-semibold">Prompt</h3>
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                            prompts.length < MIN_PROMPTS
                                ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                                : "bg-muted text-muted-foreground"
                        }`}>
                            {prompts.length} / {MAX_PROMPTS}
                        </span>
                    </div>
                    {prompts.length < MAX_PROMPTS && !showAddPrompt && (
                        <Button variant="outline" size="sm" onClick={() => setShowAddPrompt(true)}>
                            <Plus className="w-3.5 h-3.5 mr-1.5" />
                            Aggiungi prompt
                        </Button>
                    )}
                </div>

                {showAddPrompt && (
                    <div className="rounded-xl border bg-card p-4 space-y-3">
                        <div className="space-y-1.5">
                            <label className="text-sm font-medium">Domanda</label>
                            <Select value={newPromptKey} onValueChange={setNewPromptKey}>
                                <SelectTrigger>
                                    <SelectValue placeholder="Scegli un prompt…" />
                                </SelectTrigger>
                                <SelectContent>
                                    {Object.entries(PROMPTS_BY_CATEGORY).map(([category, catPrompts]) => (
                                        <SelectGroup key={category}>
                                            <SelectLabel>{CATEGORY_LABELS[category as keyof typeof CATEGORY_LABELS]}</SelectLabel>
                                            {catPrompts.map((p) => (
                                                <SelectItem
                                                    key={p.key}
                                                    value={p.key}
                                                    disabled={usedPromptKeys.has(p.key)}
                                                >
                                                    {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                                    {tPrompts(p.key as any)}
                                                    {usedPromptKeys.has(p.key) && " ✓"}
                                                </SelectItem>
                                            ))}
                                        </SelectGroup>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="space-y-1.5">
                            <label className="text-sm font-medium">La tua risposta</label>
                            <Textarea
                                placeholder="Scrivi qui…"
                                value={newPromptContent}
                                onChange={(e) => setNewPromptContent(e.target.value)}
                                rows={3}
                                maxLength={300}
                            />
                            <p className="text-xs text-muted-foreground text-right">{newPromptContent.length}/300</p>
                        </div>
                        <div className="flex gap-2 justify-end">
                            <Button variant="ghost" size="sm" onClick={() => { setShowAddPrompt(false); setNewPromptKey(""); setNewPromptContent(""); }}>
                                Annulla
                            </Button>
                            <Button size="sm" onClick={handleAddPrompt} disabled={!newPromptKey || !newPromptContent.trim()}>
                                Aggiungi
                            </Button>
                        </div>
                    </div>
                )}

                {prompts.length === 0 ? (
                    <button
                        type="button"
                        onClick={() => setShowAddPrompt(true)}
                        className="w-full rounded-xl border border-dashed py-10 text-center text-sm text-muted-foreground hover:border-foreground/30 hover:text-foreground transition-colors"
                    >
                        <MessageSquare className="w-6 h-6 mx-auto mb-2 opacity-40" />
                        Nessun prompt — aggiungine almeno {MIN_PROMPTS}
                    </button>
                ) : (
                    <DndContext
                        sensors={sensors}
                        collisionDetection={closestCenter}
                        onDragStart={handlePromptDragStart}
                        onDragEnd={handlePromptDragEnd}
                    >
                        <SortableContext items={prompts.map((p) => p._key)} strategy={verticalListSortingStrategy}>
                            <div className="space-y-3">
                                {prompts.map((prompt) => (
                                    <SortablePrompt
                                        key={prompt._key}
                                        prompt={prompt}
                                        onDelete={handleDeletePrompt}
                                        tPrompts={tPrompts}
                                    />
                                ))}
                            </div>
                        </SortableContext>
                        <DragOverlay>
                            {activePrompt && (
                                <SortablePrompt
                                    prompt={activePrompt}
                                    onDelete={() => {}}
                                    isDragOverlay
                                    tPrompts={tPrompts}
                                />
                            )}
                        </DragOverlay>
                    </DndContext>
                )}
            </section>
        </div>
    );
}

// ─── Helpers for converting to/from API types ─────────────────────────────────

export function localPhotosToInput(photos: LocalPhoto[]): PhotoItemInput[] {
    return photos.map((p, i) => ({ type: "photo" as const, content: p.content, displayOrder: i }));
}

export function localPromptsToInput(prompts: LocalPrompt[]): PromptItemInput[] {
    return prompts.map((p, i) => ({ type: "prompt" as const, promptKey: p.promptKey, content: p.content, displayOrder: i }));
}

export function userItemsToLocal(
    items: { type: string; content: string; promptKey?: string | null; displayOrder: number }[]
): { photos: LocalPhoto[]; prompts: LocalPrompt[] } {
    const sorted = [...items].sort((a, b) => a.displayOrder - b.displayOrder);
    return {
        photos: sorted
            .filter((i) => i.type === "photo")
            .map((i) => ({ _key: makeKey(), content: i.content })),
        prompts: sorted
            .filter((i) => i.type === "prompt")
            .map((i) => ({ _key: makeKey(), promptKey: i.promptKey ?? "", content: i.content })),
    };
}
