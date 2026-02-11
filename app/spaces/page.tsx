import { query } from "@/lib/graphql/apollo-client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PageShell } from "@/components/page-shell";
import { SpaceCard } from "@/components/spaces/space-card";
import { CreateSpaceButton } from "./create-space-button";
import gql from "graphql-tag";

const GET_ALL_SPACES = gql`
  query GetAllSpaces {
    spaces {
      id
      name
      slug
      description
      clientId
      isActive
      visibility
      joinPolicy
      membersCount
      createdAt
      image
    }
  }
`;

interface Space {
  id: string;
  name: string;
  slug: string;
  description?: string;
  clientId: string;
  isActive: boolean;
  visibility: string;
  joinPolicy: string;
  membersCount: number;
  createdAt: string;
  image?: string;
}

export default async function DiscoverSpacesPage() {
  const { data } = await query({ query: GET_ALL_SPACES });
  const spaces = (data as { spaces: Space[] }).spaces ?? [];

  return (
    <PageShell
      header={
        <div className="space-y-1">
          <h1 className="text-4xl font-extrabold tracking-tight">Discover Spaces</h1>
          <p className="text-lg text-muted-foreground font-medium">Explore and join communities and clubs</p>
        </div>
      }
      actions={<CreateSpaceButton />}
    >
      {spaces.length === 0 ? (
        <Card className="text-center py-12 shadow-none border-dashed">
          <CardHeader>
            <div className="text-6xl mb-4">🪐</div>
            <CardTitle className="text-xl">No spaces yet</CardTitle>
            <CardDescription>Create your first space to start building your community</CardDescription>
          </CardHeader>
          <CardContent>
            <CreateSpaceButton />
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {spaces.map((space) => (
            <SpaceCard key={space.id} space={space} />
          ))}
        </div>
      )}
    </PageShell>
  );
}
