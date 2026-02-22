import { query } from "@/lib/graphql/apollo-client";
import { GET_ALL_SPACES } from "@/lib/models/spaces/gql";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { CreateSpaceButton } from "./create-space-button";
import type { GetAllSpacesQuery } from "@/lib/graphql/__generated__/graphql";

export default async function SpacesPage() {
  const { data } = await query<GetAllSpacesQuery>({ query: GET_ALL_SPACES });
  const spaces = data?.spaces ?? [];

  return (
    <Page
      breadcrumbs={[{ label: "Spaces" }]}
      header={
        <div className="space-y-1">
          <h1 className="text-6xl font-extrabold tracking-tight">Spaces</h1>
          <p className="text-lg text-muted-foreground font-medium">Esplora le community</p>
        </div>
      }
      actions={<CreateSpaceButton />}
    >
      {spaces.length === 0 ? (
        <p className="text-muted-foreground text-sm">Nessuno space disponibile.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {spaces.map((space) => (
            <SpaceCard key={space.id} space={space} />
          ))}
        </div>
      )}
    </Page>
  );
}
