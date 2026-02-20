"use client";

import { useParams } from "next/navigation";
import { useQuery, useMutation } from "@apollo/client/react";
import Link from "next/link";
import { Page } from "@/components/page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
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
} from "lucide-react";
import { GET_EVENT, RESPOND_TO_EVENT } from "@/lib/models/events/gql";
import {
  AttendeeStatus,
  type GetEventQuery,
  type GetEventQueryVariables,
  type RespondToEventMutation,
  type RespondToEventMutationVariables,
} from "@/lib/graphql/__generated__/graphql";

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

export default function EventDetailPage() {
  const { id } = useParams<{ id: string }>();

  const { data, loading, refetch } = useQuery<GetEventQuery, GetEventQueryVariables>(GET_EVENT, {
    variables: { id },
  });

  const [respondToEvent, { loading: responding }] = useMutation<
    RespondToEventMutation,
    RespondToEventMutationVariables
  >(RESPOND_TO_EVENT, { onCompleted: () => refetch() });

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
      {/* Two-column DICE-style layout */}
      <div className="flex flex-col lg:flex-row gap-10 lg:gap-16 items-start">

        {/* ── LEFT — sticky cover ── */}
        <div className="w-full lg:w-[300px] xl:w-[340px] shrink-0 lg:sticky lg:top-24 space-y-4">
          {/* Square cover — always gradient */}
          <div className="aspect-square w-full rounded-2xl overflow-hidden ring-1 ring-border/50">
            <div className={`size-full bg-gradient-to-br ${gradient} flex items-center justify-center`}>
              <span className="text-9xl font-black text-white/20 select-none">
                {event.title.charAt(0).toUpperCase()}
              </span>
            </div>
          </div>

          {/* Back link */}
          <Button variant="ghost" size="sm" asChild className="text-muted-foreground w-full justify-start">
            <Link href="/events">
              <ArrowLeftIcon className="h-4 w-4 mr-2" />
              Tutti gli eventi
            </Link>
          </Button>
        </div>

        {/* ── RIGHT — scrollable content ── */}
        <div className="flex-1 min-w-0 space-y-8">

          {/* Title block */}
          <div className="space-y-2">
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

          {/* RSVP card */}
          {isPublished && !isPast && (
            <div className="border rounded-xl p-5 space-y-3 bg-muted/30">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div>
                  {myStatus === AttendeeStatus.Going ? (
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
                  {myStatus === AttendeeStatus.Going ? (
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

          {/* Description */}
          {event.description && (
            <div className="space-y-2">
              <h2 className="text-base font-semibold">A proposito di</h2>
              <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">
                {event.description}
              </p>
            </div>
          )}

          <Separator />

          {/* Date & time detail */}
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

          {/* Location */}
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

          {/* Tags */}
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

          {/* Participants */}
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
      </div>
    </Page>
  );
}
