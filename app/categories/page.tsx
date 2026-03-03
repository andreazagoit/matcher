import { query } from "@/lib/graphql/apollo-client";
import { GET_CATEGORIES } from "@/lib/models/categories/gql";
import { Page } from "@/components/page";
import Link from "next/link";

export default async function CategoriesPage() {
  const { data } = await query<{ categories: { id: string }[] }>({ query: GET_CATEGORIES });
  const categories = data?.categories ?? [];

  return (
    <Page
      breadcrumbs={[{ label: "Categories" }]}
      header={
        <div className="space-y-1">
          <h1 className="text-6xl font-extrabold tracking-tight">Categories</h1>
          <p className="text-lg text-muted-foreground font-medium">
            Esplora eventi e spazi per interesse
          </p>
        </div>
      }
    >
      {categories.length === 0 ? (
        <p className="text-muted-foreground text-sm">
          Nessuna categoria disponibile.
        </p>
      ) : (
        <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
          {categories.map((cat) => (
            <Link
              key={cat.id}
              href={`/categories/${cat.id}`}
              className="group flex items-center justify-center rounded-2xl border bg-card px-4 py-6 text-center transition-colors hover:bg-accent hover:border-accent-foreground/20"
            >
              <span className="text-sm font-semibold capitalize group-hover:text-accent-foreground">
                {cat.id}
              </span>
            </Link>
          ))}
        </div>
      )}
    </Page>
  );
}
