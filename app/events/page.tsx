"use client";

import { useState } from "react";
import { useQuery, useMutation } from "@apollo/client/react";
import { Page } from "@/components/page";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CalendarIcon, Loader2Icon, SparklesIcon, CalendarCheckIcon } from "lucide-react";
import { GET_MY_UPCOMING_EVENTS, GET_RECOMMENDED_EVENTS, RESPOND_TO_EVENT } from "@/lib/models/events/gql";
import {
  AttendeeStatus,
  type MyUpcomingEventsQuery,
  type RecommendedEventsQuery,
  type RespondToEventMutation,
  type RespondToEventMutationVariables,
} from "@/lib/graphql/__generated__/graphql";
import { EventCard, type EventCardEvent } from "@/components/event-card";

type EventItem = NonNullable<RecommendedEventsQuery["recommendedEvents"]>[number];

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
            <EventCard key={event.id} event={event as EventCardEvent} onRespond={handleRespond} />
          ))}
        </div>
      )}
    </Page>
  );
}
