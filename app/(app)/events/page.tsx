"use client";

import { useState } from "react";
import { useQuery, useMutation } from "@apollo/client/react";
import Link from "next/link";
import { Page } from "@/components/page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  CalendarIcon,
  MapPinIcon,
  UsersIcon,
  Loader2Icon,
  SparklesIcon,
  CalendarCheckIcon,
} from "lucide-react";
import {
  GET_MY_UPCOMING_EVENTS,
  GET_RECOMMENDED_EVENTS,
  RESPOND_TO_EVENT,
} from "@/lib/models/events/gql";
import {
  AttendeeStatus,
  type MyUpcomingEventsQuery,
  type RecommendedEventsQuery,
  type RespondToEventMutation,
  type RespondToEventMutationVariables,
} from "@/lib/graphql/__generated__/graphql";

type EventItem = NonNullable<RecommendedEventsQuery["recommendedEvents"]>[number];

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

function EventCard({ event, onRespond }: { event: EventItem; onRespond: (id: string, s: AttendeeStatus) => void }) {
  const startDate = new Date(event.startsAt as string);
  const isPast = startDate < new Date();
  const isPublished = event.status === "published";
  const isCompleted = event.status === "completed";
  const gradient = getGradient(event.tags ?? []);

  return (
    <div className={`flex flex-col gap-3 group ${isCompleted ? "opacity-60" : ""}`}>
      {/* Image — aspect-square like SpaceCard */}
      <Link href={`/events/${event.id}`} className="block">
        <div className="aspect-square w-full relative bg-muted rounded-xl overflow-hidden ring-1 ring-border/50 group-hover:ring-primary/20 transition-all">
          <div className={`size-full bg-gradient-to-br ${gradient} flex items-center justify-center`}>
            <span className="text-6xl font-black text-white/20 select-none group-hover:scale-110 transition-transform">
              {event.title.charAt(0).toUpperCase()}
            </span>
          </div>

          {/* Date badge — top right */}
          <div className="absolute top-2 right-2 flex gap-1">
            <Badge className="bg-white/90 text-black hover:bg-white shadow-sm backdrop-blur-sm h-6 gap-1 font-medium">
              <CalendarIcon className="h-3 w-3" />
              {startDate.toLocaleDateString("it-IT", { day: "numeric", month: "short" })}
            </Badge>
          </div>

          {/* Time badge — bottom left */}
          <div className="absolute bottom-2 left-2">
            <Badge className="bg-black/60 text-white hover:bg-black/70 backdrop-blur-sm h-6 font-mono">
              {startDate.toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" })}
            </Badge>
          </div>
        </div>
      </Link>

      {/* Content below — same as SpaceCard */}
      <div className="space-y-1.5 px-1">
        <h3 className="font-semibold tracking-tight text-lg leading-tight group-hover:text-primary transition-colors line-clamp-1">
          {event.title}
        </h3>

        <p className="text-sm text-muted-foreground line-clamp-1">
          {event.description || (event.location ?? "Nessuna descrizione")}
        </p>

        <div className="flex items-center gap-3 pt-1 text-xs text-muted-foreground">
          {event.location && (
            <div className="flex items-center gap-1.5 min-w-0">
              <MapPinIcon className="size-3.5 shrink-0" />
              <span className="truncate">{event.location}</span>
            </div>
          )}
          <div className="flex items-center gap-1.5 shrink-0 ml-auto">
            <UsersIcon className="size-3.5" />
            <span>{event.attendeeCount}{event.maxAttendees ? `/${event.maxAttendees}` : ""}</span>
          </div>
        </div>

        {isPublished && !isPast && (
          <div className="flex gap-2 pt-1">
            <Button size="sm" className="flex-1" onClick={() => onRespond(event.id, AttendeeStatus.Going)}>
              Partecipo
            </Button>
            <Button size="sm" variant="outline" onClick={() => onRespond(event.id, AttendeeStatus.Interested)}>
              Interessato
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function EventsPage() {
  const [tab, setTab] = useState<"recommended" | "mine">("recommended");

  const { data: recommendedData, loading: loadingRec, refetch: refetchRec } =
    useQuery<RecommendedEventsQuery>(GET_RECOMMENDED_EVENTS, {
      variables: { limit: 20 },
      skip: tab !== "recommended",
    });

  const { data: myData, loading: loadingMine, refetch: refetchMine } =
    useQuery<MyUpcomingEventsQuery>(GET_MY_UPCOMING_EVENTS, {
      skip: tab !== "mine",
    });

  const [respondToEvent] = useMutation<RespondToEventMutation, RespondToEventMutationVariables>(
    RESPOND_TO_EVENT,
    { onCompleted: () => { refetchRec(); refetchMine(); } },
  );

  const handleRespond = (eventId: string, status: AttendeeStatus) => {
    respondToEvent({ variables: { eventId, status } }).catch(console.error);
  };

  const recommended = recommendedData?.recommendedEvents ?? [];
  const mine = (myData?.myUpcomingEvents ?? []) as EventItem[];
  const loading = tab === "recommended" ? loadingRec : loadingMine;
  const events = tab === "recommended" ? recommended : mine;

  return (
    <Page
      breadcrumbs={[{ label: "Eventi" }]}
      header={
        <div className="space-y-1">
          <h1 className="text-4xl font-extrabold tracking-tight">Eventi</h1>
          <p className="text-lg text-muted-foreground font-medium">
            Scopri e partecipa agli eventi della community
          </p>
        </div>
      }
    >
      <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)}>
        <TabsList className="mb-6">
          <TabsTrigger value="recommended" className="gap-2">
            <SparklesIcon className="h-4 w-4" />
            Consigliati
          </TabsTrigger>
          <TabsTrigger value="mine" className="gap-2">
            <CalendarCheckIcon className="h-4 w-4" />
            I miei
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {loading ? (
        <div className="flex items-center justify-center py-24">
          <Loader2Icon className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : events.length === 0 ? (
        <div className="text-center py-24 bg-muted/10 rounded-2xl border-2 border-dashed border-muted-foreground/20">
          <CalendarIcon className="h-12 w-12 mx-auto mb-4 text-muted-foreground/40" />
          <h3 className="text-xl font-semibold">Nessun evento</h3>
          <p className="text-muted-foreground mt-2 max-w-sm mx-auto">
            {tab === "recommended"
              ? "Completa il tuo profilo e aggiungi interessi per ricevere suggerimenti personalizzati."
              : "Non sei iscritto a nessun evento imminente."}
          </p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {events.map((event) => (
            <EventCard key={event.id} event={event} onRespond={handleRespond} />
          ))}
        </div>
      )}
    </Page>
  );
}
