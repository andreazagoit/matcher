"use client";

import { useParams, useSearchParams, useRouter } from "next/navigation";
import { useQuery, useMutation } from "@apollo/client/react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { Page } from "@/components/page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { EventMap } from "@/components/event-map";
import {
  CalendarIcon,
  MapPinIcon,
  UsersIcon,
  Loader2Icon,
  ClockIcon,
  TagIcon,
  CheckCircle2Icon,
  StarIcon,
  LockIcon,
  ArrowLeftIcon,
  TicketIcon,
  EuroIcon,
  ShieldCheckIcon,
  SaveIcon,
  XCircleIcon,
  CheckSquareIcon,
  SendIcon,
  FileEditIcon,
} from "lucide-react";
import { GET_EVENT, RESPOND_TO_EVENT, UPDATE_EVENT } from "@/lib/models/events/gql";
import {
  AttendeeStatus,
  type GetEventQuery,
  type GetEventQueryVariables,
  type RespondToEventMutation,
  type RespondToEventMutationVariables,
} from "@/lib/graphql/__generated__/graphql";

function formatPrice(cents: number, currency: string) {
  return new Intl.NumberFormat("it-IT", {
    style: "currency",
    currency: currency.toUpperCase(),
    minimumFractionDigits: 2,
  }).format(cents / 100);
}

const TAG_GRADIENTS: Record<string, string> = {
  trekking: "from-emerald-500 to-teal-700",
  mountains: "from-slate-500 to-slate-800",
  cycling: "from-lime-500 to-green-700",
  running: "from-orange-400 to-red-600",
  yoga: "from-violet-400 to-purple-700",
  gym: "from-blue-500 to-indigo-700",
  cooking: "from-amber-400 to-orange-600",
  wine: "from-rose-500 to-red-800",
  restaurants: "from-pink-400 to-rose-600",
  music: "from-purple-500 to-violet-800",
  art: "from-fuchsia-400 to-pink-700",
  coding: "from-cyan-500 to-blue-700",
  gaming: "from-indigo-400 to-purple-600",
  parties: "from-yellow-400 to-orange-500",
  cinema: "from-neutral-500 to-neutral-800",
  reading: "from-stone-400 to-stone-700",
  travel: "from-sky-400 to-blue-600",
};

function getGradient(tags: string[]) {
  for (const tag of tags) {
    if (TAG_GRADIENTS[tag]) return TAG_GRADIENTS[tag];
  }
  return "from-primary/60 to-primary";
}

const STATUS_LABELS: Record<string, string> = {
  draft: "Bozza",
  published: "Pubblicato",
  cancelled: "Cancellato",
  completed: "Concluso",
};

const STATUS_BADGE: Record<string, string> = {
  draft: "bg-yellow-500/15 text-yellow-600 border-yellow-500/30",
  published: "bg-green-500/15 text-green-600 border-green-500/30",
  cancelled: "bg-red-500/15 text-red-600 border-red-500/30",
  completed: "bg-blue-500/15 text-blue-600 border-blue-500/30",
};

const ATTENDEE_STATUS_LABELS: Record<string, string> = {
  going: "Partecipa",
  interested: "Interessato",
  attended: "Presente",
};

const ATTENDEE_STATUS_BADGE: Record<string, string> = {
  going: "bg-green-500/15 text-green-600 border-green-500/30",
  interested: "bg-yellow-500/15 text-yellow-600 border-yellow-500/30",
  attended: "bg-blue-500/15 text-blue-600 border-blue-500/30",
};

// ─── to datetime-local input format ──────────────────────────────────────────
function toDatetimeLocal(iso: string) {
  const d = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export default function EventDetailPage() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [buyingTicket, setBuyingTicket] = useState(false);
  const [successBanner, setSuccessBanner] = useState(false);

  const { data, loading, refetch } = useQuery<GetEventQuery, GetEventQueryVariables>(GET_EVENT, {
    variables: { id },
  });

  const [respondToEvent, { loading: responding }] = useMutation<
    RespondToEventMutation,
    RespondToEventMutationVariables
  >(RESPOND_TO_EVENT, { onCompleted: () => refetch() });

  const [updateEvent, { loading: updating }] = useMutation(UPDATE_EVENT, {
    onCompleted: () => refetch(),
  });

  // Show a success banner when returning from Stripe Checkout
  useEffect(() => {
    if (searchParams.get("success") === "1") {
      setSuccessBanner(true);
      router.replace(`/events/${id}`);
      refetch();
    }
  }, [searchParams, id, router, refetch]);

  const handleBuyTicket = async () => {
    setBuyingTicket(true);
    try {
      const res = await fetch("/api/stripe/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eventId: id }),
      });
      const json = await res.json();
      if (json.url) {
        window.location.href = json.url;
      } else {
        alert(json.error ?? "Errore durante il checkout");
      }
    } catch {
      alert("Errore di rete. Riprova.");
    } finally {
      setBuyingTicket(false);
    }
  };

  const event = data?.event;

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2Icon className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!event) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <LockIcon className="h-10 w-10 text-muted-foreground" />
        <p className="font-semibold text-lg">Evento non disponibile</p>
        <p className="text-sm text-muted-foreground text-center max-w-xs">
          L&apos;evento potrebbe non esistere o appartenere a uno space privato a cui non sei iscritto.
        </p>
        <Button asChild variant="outline">
          <Link href="/events">← Torna agli eventi</Link>
        </Button>
      </div>
    );
  }

  const startDate = new Date(event.startsAt as string);
  const endDate = event.endsAt ? new Date(event.endsAt as string) : null;
  const isPast = startDate < new Date();
  const isPublished = event.status === "published";
  const gradient = getGradient(event.tags ?? []);
  const myStatus = event.myAttendeeStatus;
  const isPaid = event.isPaid;
  const myPaymentStatus = event.myPaymentStatus;
  const hasTicket = isPaid && myPaymentStatus === "paid";

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const isAdmin = (event.space as any)?.myMembership?.role === "admin";

  const dateLabel = startDate.toLocaleDateString("it-IT", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
  const timeLabel = startDate.toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" });

  return (
    <Page
      breadcrumbs={[
        { label: "Eventi", href: "/events" },
        ...(event.space ? [{ label: event.space.name, href: `/spaces/${event.space.slug}` }] : []),
        { label: event.title },
      ]}
      header={<div />}
    >
      <div className="flex flex-col lg:flex-row gap-10 lg:gap-16 items-start">

        {/* ── LEFT — sticky cover ── */}
        <div className="w-full lg:w-[300px] xl:w-[340px] shrink-0 lg:sticky lg:top-24 space-y-4">
          <div className="aspect-square w-full rounded-2xl overflow-hidden ring-1 ring-border/50">
            <div className={`size-full bg-gradient-to-br ${gradient} flex items-center justify-center`}>
              <span className="text-9xl font-black text-white/20 select-none">
                {event.title.charAt(0).toUpperCase()}
              </span>
            </div>
          </div>

          <Button variant="ghost" size="sm" asChild className="text-muted-foreground w-full justify-start">
            <Link href="/events">
              <ArrowLeftIcon className="h-4 w-4 mr-2" />
              Tutti gli eventi
            </Link>
          </Button>
        </div>

        {/* ── RIGHT — content with tabs ── */}
        <div className="flex-1 min-w-0">

          {/* Title block — always visible above tabs */}
          <div className="space-y-2 mb-8">
            {event.space && (
              <Link
                href={`/spaces/${event.space.slug}`}
                className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors inline-flex items-center gap-1.5"
              >
                {event.space.visibility !== "public" && <LockIcon className="h-3 w-3" />}
                {event.space.name}
              </Link>
            )}
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight leading-tight">
              {event.title}
            </h1>
            <p className="text-primary font-semibold text-base">
              {dateLabel}, {timeLabel}
              {endDate && (
                <span className="text-muted-foreground font-normal">
                  {" "}→ {endDate.toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" })}
                </span>
              )}
            </p>
            <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground pt-0.5">
              {event.tags && event.tags.slice(0, 3).map((tag: string) => (
                <span key={tag} className="inline-flex items-center gap-1">
                  <TagIcon className="h-3.5 w-3.5" />
                  {tag}
                </span>
              ))}
              {event.location && (
                <span className="inline-flex items-center gap-1">
                  <MapPinIcon className="h-3.5 w-3.5" />
                  {event.location}
                </span>
              )}
            </div>
          </div>

          {isAdmin ? (
            <Tabs defaultValue="details">
              <TabsList className="mb-6">
                <TabsTrigger value="details">Dettagli</TabsTrigger>
                <TabsTrigger value="manage" className="gap-1.5">
                  <ShieldCheckIcon className="h-3.5 w-3.5" />
                  Gestione
                </TabsTrigger>
              </TabsList>

              <TabsContent value="details">
                <DetailsTab
                  event={event}
                  isPast={isPast}
                  isPublished={isPublished}
                  isPaid={isPaid}
                  hasTicket={hasTicket}
                  myStatus={myStatus}
                  responding={responding}
                  buyingTicket={buyingTicket}
                  successBanner={successBanner}
                  startDate={startDate}
                  endDate={endDate}
                  timeLabel={timeLabel}
                  respondToEvent={respondToEvent}
                  handleBuyTicket={handleBuyTicket}
                />
              </TabsContent>

              <TabsContent value="manage">
                <ManageTab
                  event={event}
                  updating={updating}
                  updateEvent={updateEvent}
                  refetch={refetch}
                />
              </TabsContent>
            </Tabs>
          ) : (
            <DetailsTab
              event={event}
              isPast={isPast}
              isPublished={isPublished}
              isPaid={isPaid}
              hasTicket={hasTicket}
              myStatus={myStatus}
              responding={responding}
              buyingTicket={buyingTicket}
              successBanner={successBanner}
              startDate={startDate}
              endDate={endDate}
              timeLabel={timeLabel}
              respondToEvent={respondToEvent}
              handleBuyTicket={handleBuyTicket}
            />
          )}
        </div>
      </div>
    </Page>
  );
}

// ─── Details Tab ─────────────────────────────────────────────────────────────

function DetailsTab({
  event,
  isPast,
  isPublished,
  isPaid,
  hasTicket,
  myStatus,
  responding,
  buyingTicket,
  successBanner,
  startDate,
  endDate,
  timeLabel,
  respondToEvent,
  handleBuyTicket,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  event: any;
  isPast: boolean;
  isPublished: boolean;
  isPaid: boolean;
  hasTicket: boolean;
  myStatus: AttendeeStatus | null | undefined;
  responding: boolean;
  buyingTicket: boolean;
  successBanner: boolean;
  startDate: Date;
  endDate: Date | null;
  timeLabel: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  respondToEvent: (opts: any) => void;
  handleBuyTicket: () => void;
}) {
  return (
    <div className="space-y-8">
      {successBanner && (
        <div className="rounded-xl border border-green-500/40 bg-green-500/10 px-5 py-4 flex items-center gap-3">
          <CheckCircle2Icon className="h-5 w-5 text-green-600 shrink-0" />
          <div>
            <p className="font-semibold text-sm text-green-700 dark:text-green-400">Biglietto acquistato!</p>
            <p className="text-xs text-green-600/80 dark:text-green-500/80">Il tuo posto è confermato. A presto all&apos;evento!</p>
          </div>
        </div>
      )}

      {isPublished && !isPast && (
        <div className="border rounded-xl p-5 space-y-3 bg-muted/30">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              {isPaid ? (
                hasTicket ? (
                  <p className="font-semibold text-sm flex items-center gap-1.5 text-primary">
                    <TicketIcon className="h-4 w-4" />
                    Biglietto confermato
                  </p>
                ) : (
                  <div>
                    <p className="font-semibold text-sm flex items-center gap-1.5">
                      <EuroIcon className="h-4 w-4" />
                      Evento a pagamento
                    </p>
                    <p className="text-lg font-bold mt-0.5">
                      {formatPrice(event.price!, event.currency ?? "eur")}
                    </p>
                  </div>
                )
              ) : myStatus === AttendeeStatus.Going ? (
                <p className="font-semibold text-sm flex items-center gap-1.5 text-primary">
                  <CheckCircle2Icon className="h-4 w-4" />
                  Partecipi a questo evento
                </p>
              ) : myStatus === AttendeeStatus.Interested ? (
                <p className="font-semibold text-sm flex items-center gap-1.5">
                  <StarIcon className="h-4 w-4" />
                  Sei interessato a questo evento
                </p>
              ) : (
                <p className="font-semibold text-sm">Partecipa all&apos;evento</p>
              )}
              <p className="text-xs text-muted-foreground mt-0.5">
                {event.attendeeCount} iscritti
                {event.maxAttendees && ` · ${event.maxAttendees - event.attendeeCount} posti rimasti`}
              </p>
            </div>
            <div className="flex gap-2 flex-wrap">
              {isPaid ? (
                hasTicket ? (
                  <Badge variant="secondary" className="text-xs px-3 py-1.5">
                    <TicketIcon className="h-3 w-3 mr-1.5" />
                    Biglietto valido
                  </Badge>
                ) : (
                  <Button
                    disabled={buyingTicket || !event.space?.stripeAccountEnabled}
                    onClick={handleBuyTicket}
                  >
                    {buyingTicket
                      ? <Loader2Icon className="h-4 w-4 animate-spin mr-2" />
                      : <TicketIcon className="h-4 w-4 mr-2" />
                    }
                    {buyingTicket ? "Reindirizzamento..." : "Acquista Biglietto"}
                  </Button>
                )
              ) : myStatus === AttendeeStatus.Going ? (
                <Button
                  size="sm"
                  variant="outline"
                  disabled={responding}
                  onClick={() => respondToEvent({ variables: { eventId: event.id, status: AttendeeStatus.Interested } })}
                >
                  {responding && <Loader2Icon className="h-3.5 w-3.5 animate-spin mr-1.5" />}
                  Cambia in Interessato
                </Button>
              ) : myStatus === AttendeeStatus.Interested ? (
                <Button
                  size="sm"
                  disabled={responding}
                  onClick={() => respondToEvent({ variables: { eventId: event.id, status: AttendeeStatus.Going } })}
                >
                  {responding && <Loader2Icon className="h-3.5 w-3.5 animate-spin mr-1.5" />}
                  Confermo partecipazione
                </Button>
              ) : (
                <>
                  <Button
                    disabled={responding}
                    onClick={() => respondToEvent({ variables: { eventId: event.id, status: AttendeeStatus.Going } })}
                  >
                    {responding && <Loader2Icon className="h-4 w-4 animate-spin mr-2" />}
                    Partecipo
                  </Button>
                  <Button
                    variant="outline"
                    disabled={responding}
                    onClick={() => respondToEvent({ variables: { eventId: event.id, status: AttendeeStatus.Interested } })}
                  >
                    Sono interessato
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {isPast && (
        <Badge variant="secondary" className="text-xs">Evento terminato</Badge>
      )}

      {event.description && (
        <div className="space-y-2">
          <h2 className="text-base font-semibold">A proposito di</h2>
          <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">
            {event.description}
          </p>
        </div>
      )}

      <Separator />

      <div className="space-y-3">
        <h2 className="text-base font-semibold">Data e orario</h2>
        <div className="space-y-2 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <CalendarIcon className="h-4 w-4 text-primary shrink-0" />
            <span className="capitalize font-medium text-foreground">
              {startDate.toLocaleDateString("it-IT", {
                weekday: "long",
                day: "numeric",
                month: "long",
                year: "numeric",
              })}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <ClockIcon className="h-4 w-4 text-primary shrink-0" />
            <span>
              {timeLabel}
              {endDate && (
                <>
                  {" "}→ {endDate.toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" })}
                  <span className="text-muted-foreground ml-2">
                    ({Math.round((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60))} ore)
                  </span>
                </>
              )}
            </span>
          </div>
        </div>
      </div>

      {event.location && (
        <>
          <Separator />
          <div className="space-y-3">
            <h2 className="text-base font-semibold">Luogo</h2>
            <div className="flex items-start gap-2 text-sm">
              <MapPinIcon className="h-4 w-4 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-medium text-foreground">{event.location}</p>
                {event.coordinates && (
                  <p className="text-muted-foreground text-xs mt-0.5">
                    {event.coordinates.lat.toFixed(4)}, {event.coordinates.lon.toFixed(4)}
                  </p>
                )}
              </div>
            </div>
            {event.coordinates && (
              <div className="rounded-xl overflow-hidden ring-1 ring-border/50 h-40">
                <EventMap
                  lat={event.coordinates.lat}
                  lon={event.coordinates.lon}
                  label={event.location}
                  className="size-full"
                />
              </div>
            )}
          </div>
        </>
      )}

      {event.tags && event.tags.length > 0 && (
        <>
          <Separator />
          <div className="space-y-3">
            <h2 className="text-base font-semibold">Tag</h2>
            <div className="flex flex-wrap gap-1.5">
              {event.tags.map((tag: string) => (
                <Badge key={tag} variant="secondary">{tag}</Badge>
              ))}
            </div>
          </div>
        </>
      )}

      <Separator />
      <div className="space-y-2">
        <h2 className="text-base font-semibold">Partecipanti</h2>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <UsersIcon className="h-4 w-4 text-primary shrink-0" />
          <span>
            <span className="font-medium text-foreground">{event.attendeeCount}</span> iscritti
            {event.maxAttendees && (
              <> · <span className="font-medium text-foreground">{event.maxAttendees}</span> posti totali</>
            )}
          </span>
        </div>
      </div>
    </div>
  );
}

// ─── Manage Tab ───────────────────────────────────────────────────────────────

function ManageTab({
  event,
  updating,
  updateEvent,
  refetch,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  event: any;
  updating: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  updateEvent: (opts: any) => void;
  () : void;
  refetch: () => void;
}) {
  const [editData, setEditData] = useState({
    title: event.title ?? "",
    description: event.description ?? "",
    location: event.location ?? "",
    startsAt: toDatetimeLocal(event.startsAt),
    endsAt: event.endsAt ? toDatetimeLocal(event.endsAt) : "",
    maxAttendees: event.maxAttendees ? String(event.maxAttendees) : "",
  });
  const [saveSuccess, setSaveSuccess] = useState(false);

  const update = <K extends keyof typeof editData>(k: K, v: (typeof editData)[K]) =>
    setEditData((p) => ({ ...p, [k]: v }));

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    await updateEvent({
      variables: {
        id: event.id,
        input: {
          title: editData.title,
          description: editData.description || undefined,
          location: editData.location || undefined,
          startsAt: editData.startsAt ? new Date(editData.startsAt).toISOString() : undefined,
          endsAt: editData.endsAt ? new Date(editData.endsAt).toISOString() : undefined,
          maxAttendees: editData.maxAttendees ? parseInt(editData.maxAttendees) : undefined,
        },
      },
    });
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const handleStatusChange = (status: string) => {
    updateEvent({ variables: { id: event.id, input: { status } } });
  };

  const attendees: {
    id: string;
    userId: string;
    status: string;
    registeredAt: string;
    paymentStatus: string | null;
    user: { id: string; name: string; username?: string } | null;
  }[] = event.attendees ?? [];

  const going = attendees.filter((a) => a.status === "going").length;
  const interested = attendees.filter((a) => a.status === "interested").length;
  const attended = attendees.filter((a) => a.status === "attended").length;

  return (
    <div className="space-y-10">

      {/* ── Status ── */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <ShieldCheckIcon className="h-4 w-4 text-primary" />
          <h2 className="text-base font-semibold">Stato evento</h2>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <span className={`inline-flex items-center text-xs font-medium px-2.5 py-1 rounded-full border ${STATUS_BADGE[event.status] ?? ""}`}>
            {STATUS_LABELS[event.status] ?? event.status}
          </span>
          <div className="flex gap-2 flex-wrap">
            {event.status === "draft" && (
              <Button size="sm" onClick={() => handleStatusChange("published")} disabled={updating}>
                <SendIcon className="h-3.5 w-3.5 mr-1.5" />
                Pubblica
              </Button>
            )}
            {event.status === "published" && (
              <>
                <Button size="sm" variant="outline" onClick={() => handleStatusChange("draft")} disabled={updating}>
                  <FileEditIcon className="h-3.5 w-3.5 mr-1.5" />
                  Torna a bozza
                </Button>
                <Button size="sm" onClick={() => handleStatusChange("completed")} disabled={updating}>
                  <CheckSquareIcon className="h-3.5 w-3.5 mr-1.5" />
                  Segna come concluso
                </Button>
                <Button size="sm" variant="destructive" onClick={() => handleStatusChange("cancelled")} disabled={updating}>
                  <XCircleIcon className="h-3.5 w-3.5 mr-1.5" />
                  Cancella evento
                </Button>
              </>
            )}
            {event.status === "cancelled" && (
              <Button size="sm" variant="outline" onClick={() => handleStatusChange("draft")} disabled={updating}>
                <FileEditIcon className="h-3.5 w-3.5 mr-1.5" />
                Ripristina come bozza
              </Button>
            )}
          </div>
        </div>
      </section>

      <Separator />

      {/* ── Attendees ── */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <UsersIcon className="h-4 w-4 text-primary" />
            <h2 className="text-base font-semibold">Partecipanti</h2>
          </div>
          <div className="flex gap-3 text-xs text-muted-foreground">
            <span><span className="font-medium text-foreground">{going}</span> partecipano</span>
            <span><span className="font-medium text-foreground">{interested}</span> interessati</span>
            {attended > 0 && <span><span className="font-medium text-foreground">{attended}</span> presenti</span>}
          </div>
        </div>

        {attendees.length === 0 ? (
          <p className="text-sm text-muted-foreground py-6 text-center border rounded-xl">
            Nessun partecipante ancora.
          </p>
        ) : (
          <div className="border rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/40">
                  <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">Utente</th>
                  <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">Stato</th>
                  {event.isPaid && (
                    <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">Pagamento</th>
                  )}
                  <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">Iscritto il</th>
                </tr>
              </thead>
              <tbody>
                {attendees.map((a, i) => (
                  <tr key={a.id} className={i < attendees.length - 1 ? "border-b" : ""}>
                    <td className="px-4 py-3">
                      <div className="font-medium">
                        {a.user ? a.user.name : a.userId}
                      </div>
                      {a.user?.username && (
                        <div className="text-xs text-muted-foreground">@{a.user.username}</div>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded-full border ${ATTENDEE_STATUS_BADGE[a.status] ?? ""}`}>
                        {ATTENDEE_STATUS_LABELS[a.status] ?? a.status}
                      </span>
                    </td>
                    {event.isPaid && (
                      <td className="px-4 py-3 text-xs text-muted-foreground">
                        {a.paymentStatus === "paid" ? (
                          <span className="text-green-600 font-medium">Pagato</span>
                        ) : a.paymentStatus ? (
                          a.paymentStatus
                        ) : (
                          "—"
                        )}
                      </td>
                    )}
                    <td className="px-4 py-3 text-xs text-muted-foreground">
                      {new Date(a.registeredAt).toLocaleDateString("it-IT")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <Separator />

      {/* ── Edit form ── */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <FileEditIcon className="h-4 w-4 text-primary" />
          <h2 className="text-base font-semibold">Modifica evento</h2>
        </div>

        {saveSuccess && (
          <div className="rounded-xl border border-green-500/40 bg-green-500/10 px-4 py-3 flex items-center gap-2 text-sm text-green-700 dark:text-green-400">
            <CheckCircle2Icon className="h-4 w-4 shrink-0" />
            Modifiche salvate con successo.
          </div>
        )}

        <form onSubmit={handleSave} className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="edit-title">Titolo</Label>
            <Input
              id="edit-title"
              value={editData.title}
              onChange={(e) => update("title", e.target.value)}
              required
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="edit-description">Descrizione</Label>
            <Textarea
              id="edit-description"
              value={editData.description}
              onChange={(e) => update("description", e.target.value)}
              rows={4}
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="edit-location">Luogo</Label>
            <Input
              id="edit-location"
              value={editData.location}
              onChange={(e) => update("location", e.target.value)}
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="edit-starts">Inizio</Label>
              <Input
                id="edit-starts"
                type="datetime-local"
                value={editData.startsAt}
                onChange={(e) => update("startsAt", e.target.value)}
                className="dark:[color-scheme:dark]"
                required
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="edit-ends">Fine</Label>
              <Input
                id="edit-ends"
                type="datetime-local"
                value={editData.endsAt}
                onChange={(e) => update("endsAt", e.target.value)}
                className="dark:[color-scheme:dark]"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="edit-max">Max partecipanti</Label>
            <Input
              id="edit-max"
              type="number"
              min={1}
              value={editData.maxAttendees}
              onChange={(e) => update("maxAttendees", e.target.value)}
              placeholder="Illimitati"
            />
          </div>

          <Button type="submit" disabled={updating}>
            {updating
              ? <Loader2Icon className="h-4 w-4 animate-spin mr-2" />
              : <SaveIcon className="h-4 w-4 mr-2" />
            }
            Salva modifiche
          </Button>
        </form>
      </section>
    </div>
  );
}
