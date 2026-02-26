"use client";

import { useState } from "react";
import { useQuery, useMutation } from "@apollo/client/react";
import { Loader2, Trash2, Image as ImageIcon, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "sonner";
import { useTranslations } from "next-intl";

import {
    GET_PROFILE_ITEMS,
    ADD_PROFILE_ITEM,
    DELETE_PROFILE_ITEM,
    REORDER_PROFILE_ITEMS
} from "@/lib/models/useritems/gql";
import { PROMPTS_BY_CATEGORY, CATEGORY_LABELS } from "@/lib/models/useritems/prompts";

type UserItemType = "photo" | "prompt";

interface UserItem {
    id: string;
    userId: string;
    type: UserItemType;
    promptKey?: string | null;
    content: string;
    displayOrder: number;
}

interface QueryData {
    userItems: UserItem[];
}

export function UserItemsEditor({ userId }: { userId: string }) {
    const tPrompts = useTranslations("prompts");

    const { data, loading, refetch } = useQuery<QueryData>(GET_PROFILE_ITEMS, {
        variables: { userId },
        fetchPolicy: "network-only",
    });

    const [addItem] = useMutation(ADD_PROFILE_ITEM);
    const [deleteItem] = useMutation(DELETE_PROFILE_ITEM);
    const [reorderItems] = useMutation(REORDER_PROFILE_ITEMS);

    const [isAddPhotoOpen, setIsAddPhotoOpen] = useState(false);
    const [newPhotoUrl, setNewPhotoUrl] = useState("");

    const [isAddPromptOpen, setIsAddPromptOpen] = useState(false);
    const [newPromptKey, setNewPromptKey] = useState<string>("");
    const [newPromptContent, setNewPromptContent] = useState("");

    const items: UserItem[] = data?.userItems ? [...data.userItems].sort((a, b) => a.displayOrder - b.displayOrder) : [];

    const handleAddPhoto = async () => {
        if (!newPhotoUrl.trim()) return;
        try {
            await addItem({
                variables: {
                    input: {
                        type: "photo",
                        content: newPhotoUrl.trim(),
                        displayOrder: items.length
                    }
                }
            });
            setNewPhotoUrl("");
            setIsAddPhotoOpen(false);
            refetch();
            toast.success("Foto aggiunta");
        } catch {
            toast.error("Errore nell'aggiunta della foto");
        }
    };

    const handleAddPrompt = async () => {
        if (!newPromptKey || !newPromptContent.trim()) return;
        try {
            await addItem({
                variables: {
                    input: {
                        type: "prompt",
                        promptKey: newPromptKey,
                        content: newPromptContent.trim(),
                        displayOrder: items.length
                    }
                }
            });
            setNewPromptKey("");
            setNewPromptContent("");
            setIsAddPromptOpen(false);
            refetch();
            toast.success("Prompt aggiunto");
        } catch {
            toast.error("Errore nell'aggiunta del prompt");
        }
    };

    const handleDelete = async (id: string) => {
        try {
            await deleteItem({ variables: { itemId: id } });
            refetch();
            toast.success("Elemento rimosso");
        } catch {
            toast.error("Errore nella rimozione");
        }
    };

    const handleMoveUp = async (index: number) => {
        if (index === 0) return;
        const newItems = [...items];
        const temp = newItems[index];
        newItems[index] = newItems[index - 1];
        newItems[index - 1] = temp;

        try {
            await reorderItems({ variables: { itemIds: newItems.map(i => i.id) } });
            refetch();
        } catch {
            toast.error("Errore nel riordino");
        }
    };

    const handleMoveDown = async (index: number) => {
        if (index === items.length - 1) return;
        const newItems = [...items];
        const temp = newItems[index];
        newItems[index] = newItems[index + 1];
        newItems[index + 1] = temp;

        try {
            await reorderItems({ variables: { itemIds: newItems.map(i => i.id) } });
            refetch();
        } catch {
            toast.error("Errore nel riordino");
        }
    };


    if (loading) return <div className="flex justify-center p-8"><Loader2 className="w-6 h-6 animate-spin text-muted-foreground" /></div>;

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-xl font-semibold">Foto e Prompt</h3>
                    <p className="text-sm text-muted-foreground mt-1">Gestisci i contenuti che appaiono sul tuo profilo. L&apos;ordine in cui li vedi qui è lo stesso con cui verranno mostrati agli altri utenti.</p>
                </div>
                <div className="flex gap-2">
                    <Dialog open={isAddPhotoOpen} onOpenChange={setIsAddPhotoOpen}>
                        <DialogTrigger asChild>
                            <Button variant="outline" size="sm">
                                <ImageIcon className="w-4 h-4 mr-2" />
                                Aggiungi Foto
                            </Button>
                        </DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>Aggiungi una nuova foto</DialogTitle>
                            </DialogHeader>
                            <div className="space-y-4 pt-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">URL Immagine</label>
                                    <Input placeholder="https://..." value={newPhotoUrl} onChange={e => setNewPhotoUrl(e.target.value)} />
                                    <p className="text-xs text-muted-foreground">Per ora supportiamo solo l&apos;inserimento manuale di URL esterni.</p>
                                </div>
                                <Button onClick={handleAddPhoto} className="w-full" disabled={!newPhotoUrl}>Aggiungi</Button>
                            </div>
                        </DialogContent>
                    </Dialog>

                    <Dialog open={isAddPromptOpen} onOpenChange={setIsAddPromptOpen}>
                        <DialogTrigger asChild>
                            <Button variant="outline" size="sm">
                                <MessageSquare className="w-4 h-4 mr-2" />
                                Aggiungi Prompt
                            </Button>
                        </DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>Aggiungi un prompt</DialogTitle>
                            </DialogHeader>
                            <div className="space-y-4 pt-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Seleziona la domanda</label>
                                    <Select value={newPromptKey} onValueChange={setNewPromptKey}>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Scegli un prompt..." />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {Object.entries(PROMPTS_BY_CATEGORY).map(([category, prompts]) => (
                                                <SelectGroup key={category}>
                                                    <SelectLabel>{CATEGORY_LABELS[category as keyof typeof CATEGORY_LABELS]}</SelectLabel>
                                                    {prompts.map(p => (
                                                        <SelectItem key={p.key} value={p.key}>
                                                            {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                                            {tPrompts(p.key as any)}
                                                        </SelectItem>
                                                    ))}
                                                </SelectGroup>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">La tua risposta</label>
                                    <Textarea
                                        placeholder="Scrivi qui la tua risposta..."
                                        value={newPromptContent}
                                        onChange={e => setNewPromptContent(e.target.value)}
                                        rows={4}
                                    />
                                </div>
                                <Button onClick={handleAddPrompt} className="w-full" disabled={!newPromptKey || !newPromptContent}>Aggiungi</Button>
                            </div>
                        </DialogContent>
                    </Dialog>
                </div>
            </div>

            <div className="space-y-3">
                {items.length === 0 ? (
                    <div className="rounded-xl border border-dashed py-12 text-center text-muted-foreground">
                        Non hai ancora aggiunto contenuti. Inizia aggiungendo una foto o un prompt per far conoscere la tua personalità!
                    </div>
                ) : (
                    items.map((item, index) => (
                        <div key={item.id} className="flex items-start gap-4 p-4 rounded-xl border bg-card">
                            <div className="flex flex-col gap-1 mt-1 opacity-50 hover:opacity-100 transition-opacity">
                                <button disabled={index === 0} onClick={() => handleMoveUp(index)} className="p-1 hover:bg-muted rounded text-xs disabled:opacity-20">▲</button>
                                <button disabled={index === items.length - 1} onClick={() => handleMoveDown(index)} className="p-1 hover:bg-muted rounded text-xs disabled:opacity-20">▼</button>
                            </div>

                            <div className="flex-1 min-w-0">
                                {item.type === "photo" ? (
                                    <div className="flex gap-4">
                                        <div className="w-24 h-24 rounded-lg bg-muted flex-shrink-0 overflow-hidden relative border">
                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                            <img src={item.content} alt="Profile photo" className="w-full h-full object-cover" />
                                        </div>
                                        <div className="flex flex-col justify-center">
                                            <span className="text-sm font-medium bg-muted px-2 py-1 rounded-md mb-2 w-fit">📷 Fotografia</span>
                                            <p className="text-xs text-muted-foreground truncate max-w-[200px]">{item.content}</p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="space-y-1">
                                        <span className="text-sm font-medium bg-muted px-2 py-1 rounded-md text-primary w-fit inline-flex items-center gap-1.5 mb-2">
                                            <MessageSquare className="w-3.5 h-3.5" />
                                            {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                            {item.promptKey ? tPrompts(item.promptKey as any) : "Prompt sconosciuto"}
                                        </span>
                                        <p className="text-current leading-relaxed">{item.content}</p>
                                    </div>
                                )}
                            </div>

                            <Button variant="ghost" size="icon" className="text-destructive hover:bg-destructive/10 hover:text-destructive flex-shrink-0" onClick={() => handleDelete(item.id)}>
                                <Trash2 className="w-4 h-4" />
                            </Button>
                        </div>
                    ))
                )}
            </div>
        </div >
    );
}
