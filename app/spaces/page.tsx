import { query } from "@/lib/graphql/apollo-client";
import { GET_RECOMMENDED_SPACES } from "@/lib/models/users/gql";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { CreateSpaceButton } from "./create-space-button";
import type {
  GetRecommendedSpacesQuery,
  GetRecommendedSpacesQueryVariables,
} from "@/lib/graphql/__generated__/graphql";

export default async function SpacesPage() {
  const res = await query<GetRecommendedSpacesQuery, GetRecommendedSpacesQueryVariables>({
    query: GET_RECOMMENDED_SPACES,
    variables: { limit: 24 },
  });
  const spaces = res.data?.recommendedSpaces?.nodes ?? [];

  return (
    <Page
      breadcrumbs={[{ label: "Spaces" }]}
      header={
        <div className="space-y-1">
          <h1 className="text-6xl font-extrabold tracking-tight">Spaces</h1>
          <p className="text-lg text-muted-foreground font-medium">
            Esplora le community
          </p>
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
