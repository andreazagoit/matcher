import { headers, cookies } from "next/headers";
import { auth } from "@/lib/auth";
import { query } from "@/lib/graphql/apollo-client";
import { GET_DAILY_MATCHES } from "@/lib/models/matches/gql";
import {
  GET_RECOMMENDED_CATEGORIES_WITH_EVENTS,
  GET_RECOMMENDED_SPACES,
  GET_RECOMMENDED_EVENTS,
} from "@/lib/models/users/gql";
import { GET_ALL_EVENTS } from "@/lib/models/events/gql";
import { GET_ALL_SPACES } from "@/lib/models/spaces/gql";
import type {
  GetDailyMatchesQuery,
  GetRecommendedSpacesQuery,
  GetRecommendedSpacesQueryVariables,
  GetRecommendedEventsQuery,
  GetRecommendedEventsQueryVariables,
  GetRecommendedCategoriesWithEventsQuery,
  GetRecommendedCategoriesWithEventsQueryVariables,
  GetAllEventsQuery,
  GetAllEventsQueryVariables,
  GetAllSpacesQuery,
  GetAllSpacesQueryVariables,
  SpaceFieldsFragment,
  EventCardFieldsFragment,
} from "@/lib/graphql/__generated__/graphql";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { UserCard } from "@/components/user-card";
import { LocationSelector } from "@/components/location-selector";
import { LocationBanner } from "@/components/location-banner";
import { ItemCarousel } from "@/components/item-carousel";
import { EventCard } from "@/components/event-card";
import { Card, CardContent } from "@/components/ui/card";

export default async function DiscoverPage() {
  const [cookieStore, session] = await Promise.all([
    cookies(),
    auth.api.getSession({ headers: await headers() }).catch(() => null),
  ]);

  const isAuthenticated = !!session?.user;
  const hasLocation = !!cookieStore.get("matcher_lat")?.value;

  if (isAuthenticated) {
    const [matchesRes, spacesRes, eventsRes, categoriesRes] = await Promise.all([
      query<GetDailyMatchesQuery>({ query: GET_DAILY_MATCHES })
        .catch(() => ({ data: null })),
      query<GetRecommendedSpacesQuery, GetRecommendedSpacesQueryVariables>({
        query: GET_RECOMMENDED_SPACES,
        variables: { limit: 8 },
      }).catch(() => ({ data: null })),
      query<GetRecommendedEventsQuery, GetRecommendedEventsQueryVariables>({
        query: GET_RECOMMENDED_EVENTS,
        variables: { limit: 8 },
      }).catch(() => ({ data: null })),
      query<GetRecommendedCategoriesWithEventsQuery, GetRecommendedCategoriesWithEventsQueryVariables>({
        query: GET_RECOMMENDED_CATEGORIES_WITH_EVENTS,
        variables: { limit: 6 },
      }).catch(() => ({ data: null })),
    ]);

    const matches = matchesRes.data?.me?.dailyMatches ?? [];
    const spaces = (spacesRes.data?.recommendedSpaces?.nodes ?? []) as SpaceFieldsFragment[];
    const events = (eventsRes.data?.recommendedEvents?.nodes ?? []) as EventCardFieldsFragment[];
    const categories = (categoriesRes.data?.recommendedCategories ?? [])
      .filter((cat) => cat.recommendedEvents.length > 0);

    return (
      <Page
        breadcrumbs={[{ label: "Discover" }]}
        header={
          <div className="space-y-1">
            <h1 className="text-6xl font-extrabold tracking-tight">Discover</h1>
            <p className="text-lg text-muted-foreground font-medium">
              Persone, eventi e spazi consigliati per te
            </p>
          </div>
        }
        headerExtras={<LocationSelector />}
      >
        <div className="space-y-12">
          {!hasLocation && <LocationBanner />}

          {matches.length === 0 ? (
            <Card className="border-dashed py-8 bg-muted/30">
              <CardContent className="flex flex-col items-center justify-center text-center space-y-2 py-4">
                <p className="text-muted-foreground font-medium">Nessun match disponibile oggi.</p>
                <p className="text-xs text-muted-foreground italic">
                  Imposta la tua posizione o torna più tardi.
                </p>
              </CardContent>
            </Card>
          ) : (
            <ItemCarousel title="Daily Matches" columns={4}>
              {matches.map((match) => (
                <UserCard key={match.user.id} user={match.user} compatibility={match.score} />
              ))}
            </ItemCarousel>
          )}

          {spaces.length > 0 && (
            <ItemCarousel title="Spazi consigliati" titleHref="/spaces" columns={4}>
              {spaces.map((space) => (
                <SpaceCard key={space.id} space={space} />
              ))}
            </ItemCarousel>
          )}

          {events.length > 0 && (
            <ItemCarousel title="Eventi consigliati" titleHref="/events" columns={4}>
              {events.map((event) => (
                <EventCard key={event.id} event={event} />
              ))}
            </ItemCarousel>
          )}

          {categories.map((cat) => (
            <ItemCarousel
              key={cat.id}
              title={cat.id.charAt(0).toUpperCase() + cat.id.slice(1)}
              titleHref={`/categories/${cat.id}`}
              columns={4}
            >
              {cat.recommendedEvents.map((event) => (
                <EventCard key={event.id} event={event as EventCardFieldsFragment} />
              ))}
            </ItemCarousel>
          ))}
        </div>
      </Page>
    );
  }

  // ── Unauthenticated: public showcase ────────────────────────────────────
  const [spacesRes, eventsRes] = await Promise.all([
    query<GetAllSpacesQuery, GetAllSpacesQueryVariables>({
      query: GET_ALL_SPACES,
      variables: { limit: 8 },
    }).catch(() => ({ data: null })),
    query<GetAllEventsQuery, GetAllEventsQueryVariables>({
      query: GET_ALL_EVENTS,
      variables: { limit: 8 },
    }).catch(() => ({ data: null })),
  ]);

  const spaces = (spacesRes.data?.spaces?.nodes ?? []) as SpaceFieldsFragment[];
  const events = (eventsRes.data?.events?.nodes ?? []) as EventCardFieldsFragment[];

  return (
    <Page
      breadcrumbs={[{ label: "Discover" }]}
      header={
        <div className="space-y-1">
          <h1 className="text-6xl font-extrabold tracking-tight">Discover</h1>
          <p className="text-lg text-muted-foreground font-medium">
            Esplora la community
          </p>
        </div>
      }
    >
      <div className="space-y-12">
        {spaces.length > 0 && (
          <ItemCarousel title="Spazi popolari" titleHref="/spaces" columns={4}>
            {spaces.map((space) => (
              <SpaceCard key={space.id} space={space} />
            ))}
          </ItemCarousel>
        )}

        {events.length > 0 && (
          <ItemCarousel title="Prossimi eventi" titleHref="/events" columns={4}>
            {events.map((event) => (
              <EventCard key={event.id} event={event} />
            ))}
          </ItemCarousel>
        )}
      </div>
    </Page>
  );
}
