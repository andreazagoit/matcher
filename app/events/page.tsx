import { query } from "@/lib/graphql/apollo-client";
import { Page } from "@/components/page";
import { CalendarIcon } from "lucide-react";
import { GET_RECOMMENDED_EVENTS } from "@/lib/models/users/gql";
import type {
  GetRecommendedEventsQuery,
  GetRecommendedEventsQueryVariables,
} from "@/lib/graphql/__generated__/graphql";
import { EventCard } from "@/components/event-card";

export default async function EventsPage() {
  const res = await query<GetRecommendedEventsQuery, GetRecommendedEventsQueryVariables>({
    query: GET_RECOMMENDED_EVENTS,
    variables: { limit: 24 },
  });
  const events = res.data?.recommendedEvents?.nodes ?? [];

  return (
    <Page
      breadcrumbs={[{ label: "Eventi" }]}
      header={
        <div className="space-y-1">
          <h1 className="text-4xl font-extrabold tracking-tight">Eventi</h1>
          <p className="text-lg text-muted-foreground font-medium">
            Prossimi eventi in community
          </p>
        </div>
      }
    >
      {events.length === 0 ? (
        <div className="text-center py-24 bg-muted/10 rounded-2xl border-2 border-dashed border-muted-foreground/20">
          <CalendarIcon className="h-12 w-12 mx-auto mb-4 text-muted-foreground/40" />
          <h3 className="text-xl font-semibold">Nessun evento</h3>
          <p className="text-muted-foreground mt-2 max-w-sm mx-auto">
            Non ci sono eventi in programma al momento.
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
