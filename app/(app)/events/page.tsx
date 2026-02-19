"use client";

import { useState } from "react";
import { useQuery, useMutation } from "@apollo/client/react";
import Link from "next/link";
import { Page } from "@/components/page";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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

function EventCard({ event, onRespond }: { event: EventItem; onRespond: (id: string, s: AttendeeStatus) => void }) {
  const startDate = new Date(event.startsAt as string);
  const isPast = startDate < new Date();
  const isPublished = event.status === "published";
  const isCompleted = event.status === "completed";

  return (
    <Card className={isCompleted ? "opacity-60" : "hover:border-primary/40 transition-colors"}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-base leading-snug">{event.title}</CardTitle>
            {event.description && (
              <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                {event.description}
              </p>
            )}
          </div>
          <Badge
            variant={isCompleted ? "secondary" : isPublished ? "default" : "outline"}
            className="shrink-0"
          >
            {isCompleted ? "Completato" : isPublished ? "Aperto" : event.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-3 text-sm text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <CalendarIcon className="h-3.5 w-3.5" />
            {startDate.toLocaleDateString("it-IT", {
              weekday: "short",
              day: "numeric",
              month: "short",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
          {event.location && (
            <span className="flex items-center gap-1.5">
              <MapPinIcon className="h-3.5 w-3.5" />
              {event.location}
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <UsersIcon className="h-3.5 w-3.5" />
            {event.attendeeCount} partecipanti
            {event.maxAttendees && ` / ${event.maxAttendees}`}
          </span>
        </div>

        {event.tags && event.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {event.tags.map((tag: string) => (
              <Badge key={tag} variant="secondary" className="text-xs">
                {tag}
              </Badge>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2 pt-1">
          {isPublished && !isPast && (
            <>
              <Button size="sm" onClick={() => onRespond(event.id, AttendeeStatus.Going)}>
                Partecipo
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onRespond(event.id, AttendeeStatus.Interested)}
              >
                Interessato
              </Button>
            </>
          )}
          <Button size="sm" variant="ghost" asChild className="ml-auto">
            <Link href={`/spaces/${event.spaceId}`}>Vai allo space</Link>
          </Button>
        </div>
      </CardContent>
    </Card>
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
        <div className="grid gap-4 md:grid-cols-2">
          {events.map((event) => (
            <EventCard key={event.id} event={event} onRespond={handleRespond} />
          ))}
        </div>
      )}
    </Page>
  );
}
