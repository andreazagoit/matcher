import { notFound } from "next/navigation";
import { cookies, headers } from "next/headers";
import { query } from "@/lib/graphql/apollo-client";
import { GET_CATEGORY } from "@/lib/models/categories/gql";
import { GET_EVENTS_BY_CATEGORIES } from "@/lib/models/events/gql";
import { GET_SPACES_BY_CATEGORIES } from "@/lib/models/spaces/gql";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { ItemCarousel } from "@/components/item-carousel";
import { auth } from "@/lib/auth";
import { recordImpression } from "@/lib/models/impressions/operations";
import Link from "next/link";
import { CalendarIcon, MapPinIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type {
  GetCategoryQuery,
  GetCategoryQueryVariables,
  GetEventsByCategoriesQuery,
  GetEventsByCategoriesQueryVariables,
  GetSpacesByCategoriesQuery,
  GetSpacesByCategoriesQueryVariables,
} from "@/lib/graphql/__generated__/graphql";

const CATEGORY_ICONS: Record<string, string> = {
  sport: "🏃",
  outdoor: "🏕️",
  music: "🎵",
  art: "🎨",
  food: "🍽️",
  travel: "✈️",
  wellness: "🧘",
  tech: "💻",
  culture: "🏛️",
  cinema: "🎬",
  social: "🤝",
  animals: "🐾",
  fashion: "👗",
  sustainability: "🌱",
  entrepreneurship: "🚀",
  science: "🔬",
  spirituality: "🕊️",
  volunteering: "❤️",
  nightlife: "🌙",
  photography: "📷",
  dance: "💃",
  crafts: "🪡",
  languages: "🗣️",
  comedy: "😂",
};

interface Props {
  params: Promise<{ categoryId: string }>;
}

export default async function CategoryPage({ params }: Props) {
  const { categoryId } = await params;

  const [categoryRes, eventsRes, spacesRes] = await Promise.all([
    query<GetCategoryQuery, GetCategoryQueryVariables>({
      query: GET_CATEGORY,
      variables: { id: categoryId },
    }),
    query<GetEventsByCategoriesQuery, GetEventsByCategoriesQueryVariables>({
      query: GET_EVENTS_BY_CATEGORIES,
      variables: { categories: [categoryId] },
    }).catch(() => ({ data: { eventsByCategories: [] } })),
    query<GetSpacesByCategoriesQuery, GetSpacesByCategoriesQueryVariables>({
      query: GET_SPACES_BY_CATEGORIES,
      variables: { categories: [categoryId] },
    }).catch(() => ({ data: { spacesByCategories: [] } })),
  ]);

  const category = categoryRes.data?.category;
  if (!category) notFound();

  // Track the visit server-side (fire-and-forget)
  const session = await auth.api
    .getSession({ headers: await headers() })
    .catch(() => null);
  if (session?.user) {
    recordImpression(session.user.id, categoryId, "category", "viewed");
  }

  const events = eventsRes.data?.eventsByCategories ?? [];
  const spaces = spacesRes.data?.spacesByCategories ?? [];

  const icon = CATEGORY_ICONS[category.id] ?? "🏷️";

  return (
    <Page
      breadcrumbs={[
        { label: "Categories", href: "/categories" },
        { label: category.name },
      ]}
      header={
        <div className="flex items-center gap-4">
          <span className="text-5xl">{icon}</span>
          <div className="space-y-1">
            <h1 className="text-5xl font-extrabold tracking-tight capitalize">
              {category.name}
            </h1>
            <p className="text-lg text-muted-foreground font-medium">
              {events.length} eventi · {spaces.length} spazi
            </p>
          </div>
        </div>
      }
    >
      <div className="space-y-12">
        {events.length > 0 && (
          <ItemCarousel title="Eventi">
            {events.map((event) => (
              <Link key={event.id} href={`/spaces/${event.spaceId}/events/${event.id}`}>
                <Card className="w-64 shrink-0 hover:bg-accent transition-colors">
                  <CardContent className="p-4 space-y-2">
                    <p className="font-semibold line-clamp-2 text-sm">{event.title}</p>
                    {event.location && (
                      <p className="flex items-center gap-1 text-xs text-muted-foreground">
                        <MapPinIcon className="w-3 h-3" />
                        {event.location}
                      </p>
                    )}
                    <p className="flex items-center gap-1 text-xs text-muted-foreground">
                      <CalendarIcon className="w-3 h-3" />
                      {new Date(event.startsAt).toLocaleDateString("it-IT", {
                        day: "numeric",
                        month: "short",
                        year: "numeric",
                      })}
                    </p>
                    {event.isPaid && event.price != null && (
                      <Badge variant="secondary" className="text-xs">
                        {(event.price / 100).toFixed(2)} {event.currency?.toUpperCase() ?? "EUR"}
                      </Badge>
                    )}
                  </CardContent>
                </Card>
              </Link>
            ))}
          </ItemCarousel>
        )}

        {spaces.length > 0 && (
          <section className="space-y-4">
            <h2 className="text-xl font-semibold tracking-tight">Spazi</h2>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {spaces.map((space) => (
                <SpaceCard key={space.id} space={space} />
              ))}
            </div>
          </section>
        )}

        {events.length === 0 && spaces.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <span className="text-6xl mb-4">{icon}</span>
            <p className="text-muted-foreground font-medium">
              Nessun contenuto disponibile per questa categoria.
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Torna presto — i nuovi eventi e spazi appariranno qui.
            </p>
          </div>
        )}
      </div>
    </Page>
  );
}
