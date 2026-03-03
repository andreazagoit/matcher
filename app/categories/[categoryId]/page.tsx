import { notFound } from "next/navigation";
import { headers } from "next/headers";
import { query } from "@/lib/graphql/apollo-client";
import { GET_CATEGORY } from "@/lib/models/categories/gql";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { ItemCarousel } from "@/components/item-carousel";
import { auth } from "@/lib/auth";
import { recordImpression } from "@/lib/models/impressions/operations";
import Link from "next/link";
import { CalendarIcon, MapPinIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface CategoryEvent {
  id: string;
  title: string;
  location?: string | null;
  startsAt: string;
  spaceId: string;
  price?: number | null;
  currency?: string | null;
  isPaid: boolean;
  attendeeCount: number;
}

interface CategorySpace {
  id: string;
  name: string;
  slug: string;
  description?: string | null;
  image?: string | null;
  categories: string[];
  visibility: string;
  joinPolicy: string;
  createdAt: string;
  isActive?: boolean | null;
  membersCount?: number | null;
  type?: string | null;
  stripeAccountEnabled: boolean;
}

interface Props {
  params: Promise<{ categoryId: string }>;
}

export default async function CategoryPage({ params }: Props) {
  const { categoryId } = await params;

  const res = await query<{
    category: {
      id: string;
      recommendedEvents: CategoryEvent[];
      recommendedSpaces: CategorySpace[];
      recommendedCategories: string[];
    } | null;
  }>({
    query: GET_CATEGORY,
    variables: { id: categoryId, eventsLimit: 20, spacesLimit: 20 },
  });

  const category = res.data?.category;
  if (!category) notFound();

  // Track the visit server-side (fire-and-forget)
  const session = await auth.api
    .getSession({ headers: await headers() })
    .catch(() => null);
  if (session?.user) {
    recordImpression(session.user.id, categoryId, "category", "viewed");
  }

  const events = category.recommendedEvents ?? [];
  const spaces = category.recommendedSpaces ?? [];
  const similar = category.recommendedCategories ?? [];

  return (
    <Page
      breadcrumbs={[
        { label: "Categories", href: "/categories" },
        { label: category.id },
      ]}
      header={
        <div className="space-y-1">
          <h1 className="text-5xl font-extrabold tracking-tight capitalize">
            {category.id}
          </h1>
          <p className="text-lg text-muted-foreground font-medium">
            {events.length} eventi · {spaces.length} spazi
          </p>
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

        {similar.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-xl font-semibold tracking-tight">Categorie simili</h2>
            <div className="flex flex-wrap gap-2">
              {similar.map((cat) => (
                <Link
                  key={cat}
                  href={`/categories/${cat}`}
                  className="inline-flex items-center rounded-full border bg-card px-4 py-1.5 text-sm font-medium capitalize hover:bg-accent transition-colors"
                >
                  {cat}
                </Link>
              ))}
            </div>
          </section>
        )}

        {events.length === 0 && spaces.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
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
