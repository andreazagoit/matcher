import Link from "next/link";
import { cookies } from "next/headers";
import { query } from "@/lib/graphql/apollo-client";
import { GET_FIND_MATCHES } from "@/lib/models/matches/gql";
import { Page } from "@/components/page";
import { UserCard } from "@/components/user-card";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeftIcon, ArrowRightIcon, UsersIcon } from "lucide-react";
import type { GetFindMatchesQuery, GetFindMatchesQueryVariables } from "@/lib/graphql/__generated__/graphql";

const UNLIMITED_RADIUS_KM = 1_000_000;
const PAGE_SIZE = 12;

function parsePage(value: string | undefined): number {
  const parsed = Number(value ?? "1");
  if (!Number.isFinite(parsed) || parsed < 1) return 1;
  return Math.floor(parsed);
}

export default async function UsersPage({
  searchParams,
}: {
  searchParams: Promise<{ page?: string }>;
}) {
  const params = await searchParams;
  const page = parsePage(params.page);
  const offset = (page - 1) * PAGE_SIZE;
  await cookies();

  const response = await query<GetFindMatchesQuery, GetFindMatchesQueryVariables>({
    query: GET_FIND_MATCHES,
    variables: {
      maxDistance: UNLIMITED_RADIUS_KM,
      limit: PAGE_SIZE + 1, // fetch one extra to detect next page
      offset,
    },
  }).catch(() => ({ data: { findMatches: [] as GetFindMatchesQuery["findMatches"] } }));

  const allMatches = response.data?.findMatches ?? [];
  const hasNextPage = allMatches.length > PAGE_SIZE;
  const matches = hasNextPage ? allMatches.slice(0, PAGE_SIZE) : allMatches;
  const hasPrevPage = page > 1;

  return (
    <Page
      breadcrumbs={[{ label: "Users" }]}
      header={(
        <div className="space-y-1">
          <h1 className="text-4xl font-extrabold tracking-tight">Utenti consigliati</h1>
          <p className="text-muted-foreground">
            Persone simili a te senza limiti di distanza
          </p>
        </div>
      )}
    >
      {matches.length === 0 ? (
        <Card className="border-dashed py-10 bg-muted/20">
          <CardContent className="flex flex-col items-center justify-center text-center gap-3">
            <UsersIcon className="h-10 w-10 text-muted-foreground/50" />
            <p className="font-medium">Nessun utente consigliato trovato</p>
            <p className="text-sm text-muted-foreground">
              Prova ad aumentare il raggio o completa il profilo per migliorare i suggerimenti.
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {matches.map((match) => (
              <UserCard key={match.user.id} user={match.user} compatibility={match.score} />
            ))}
          </div>

          <div className="mt-8 flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              Pagina {page}
            </div>
            <div className="flex items-center gap-2">
              {hasPrevPage ? (
                <Button asChild variant="outline" size="sm">
                  <Link href={`/users?page=${page - 1}`}>
                    <ArrowLeftIcon className="h-4 w-4 mr-1.5" />
                    Precedente
                  </Link>
                </Button>
              ) : (
                <Button variant="outline" size="sm" disabled>
                  <ArrowLeftIcon className="h-4 w-4 mr-1.5" />
                  Precedente
                </Button>
              )}

              {hasNextPage ? (
                <Button asChild size="sm">
                  <Link href={`/users?page=${page + 1}`}>
                    Successiva
                    <ArrowRightIcon className="h-4 w-4 ml-1.5" />
                  </Link>
                </Button>
              ) : (
                <Button size="sm" disabled>
                  Successiva
                  <ArrowRightIcon className="h-4 w-4 ml-1.5" />
                </Button>
              )}
            </div>
          </div>
        </>
      )}
    </Page>
  );
}
