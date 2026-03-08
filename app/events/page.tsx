import { headers } from "next/headers";
import { auth } from "@/lib/auth";
import { query } from "@/lib/graphql/apollo-client";
import { Page } from "@/components/page";
import { CalendarIcon } from "lucide-react";
import { GET_ALL_EVENTS } from "@/lib/models/events/gql";
import { GET_USER_RECOMMENDED_EVENTS } from "@/lib/models/users/gql";
import type {
  GetAllEventsQuery,
  GetAllEventsQueryVariables,
  GetUserRecommendedEventsQuery,
  GetUserRecommendedEventsQueryVariables,
} from "@/lib/graphql/__generated__/graphql";
import { EventCard } from "@/components/event-card";

export default async function EventsPage() {
  const session = await auth.api
    .getSession({ headers: await headers() })
    .catch(() => null);
  const isAuthenticated = !!session?.user;

  const events = isAuthenticated
    ? await query<GetUserRecommendedEventsQuery, GetUserRecommendedEventsQueryVariables>({
        query: GET_USER_RECOMMENDED_EVENTS,
        variables: { limit: 24 },
      }).then((res) => res.data?.me?.recommendedEvents ?? [])
    : await query<GetAllEventsQuery, GetAllEventsQueryVariables>({
        query: GET_ALL_EVENTS,
        variables: { limit: 24 },
      }).then((res) => res.data?.events ?? []);

  return (
    <Page
      breadcrumbs={[{ label: "Eventi" }]}
      header={
        <div className="space-y-1">
          <h1 className="text-4xl font-extrabold tracking-tight">Eventi</h1>
          <p className="text-lg text-muted-foreground font-medium">
            {isAuthenticated ? "Consigliati per te" : "Prossimi eventi in community"}
          </p>
        </div>
      }
    >
      {events.length === 0 ? (
        <div className="text-center py-24 bg-muted/10 rounded-2xl border-2 border-dashed border-muted-foreground/20">
          <CalendarIcon className="h-12 w-12 mx-auto mb-4 text-muted-foreground/40" />
          <h3 className="text-xl font-semibold">Nessun evento</h3>
          <p className="text-muted-foreground mt-2 max-w-sm mx-auto">
            {isAuthenticated
              ? "Completa il tuo profilo e aggiungi interessi per ricevere suggerimenti personalizzati."
              : "Non ci sono eventi in programma al momento."}
          </p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {events.map((event) => (
            <EventCard key={event.id} event={event} />
          ))}
        </div>
      )}
    </Page>
  );
}
