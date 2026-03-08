"use client";

import Link from "next/link";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ItemCarousel } from "@/components/item-carousel";
import { EventCard } from "@/components/event-card";
import { SpaceCard } from "@/components/spaces/space-card";
import type {
  EventCardFieldsFragment,
  SpaceFieldsFragment,
} from "@/lib/graphql/__generated__/graphql";

interface Props {
  categoryId: string;
  events: EventCardFieldsFragment[];
  spaces: SpaceFieldsFragment[];
  similar: string[];
}

export function CategoryContent({ categoryId, events, spaces, similar }: Props) {
  const defaultTab = events.length > 0 ? "events" : "spaces";

  return (
    <div className="space-y-6">
      {similar.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium text-muted-foreground">Categorie simili</p>
          <div className="flex flex-wrap gap-2">
            {similar.map((cat) => (
              <Link
                key={cat}
                href={`/categories/${cat}`}
                className="inline-flex items-center rounded-full border bg-card px-3 py-1 text-xs font-medium capitalize hover:bg-accent transition-colors"
              >
                {cat}
              </Link>
            ))}
          </div>
        </div>
      )}

      {(events.length > 0 || spaces.length > 0) ? (
        <Tabs defaultValue={defaultTab}>
          <TabsList>
            <TabsTrigger value="events" disabled={events.length === 0}>
              Eventi
            </TabsTrigger>
            <TabsTrigger value="spaces" disabled={spaces.length === 0}>
              Spazi
            </TabsTrigger>
          </TabsList>

          <TabsContent value="events" className="mt-6">
            <ItemCarousel columns={4}>
              {events.map((event) => (
                <EventCard key={event.id} event={event} />
              ))}
            </ItemCarousel>
          </TabsContent>

          <TabsContent value="spaces" className="mt-6">
            <ItemCarousel columns={4}>
              {spaces.map((space) => (
                <SpaceCard key={space.id} space={space} />
              ))}
            </ItemCarousel>
          </TabsContent>
        </Tabs>
      ) : (
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
  );
}
