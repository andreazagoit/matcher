import { notFound } from "next/navigation";
import { headers } from "next/headers";
import { query } from "@/lib/graphql/apollo-client";
import { GET_CATEGORY } from "@/lib/models/categories/gql";
import { Page } from "@/components/page";
import { auth } from "@/lib/auth";
import { recordImpression } from "@/lib/models/impressions/operations";
import { CategoryContent } from "./category-content";
import type { GetCategoryQuery, GetCategoryQueryVariables } from "@/lib/graphql/__generated__/graphql";

interface Props {
  params: Promise<{ categoryId: string }>;
}

export default async function CategoryPage({ params }: Props) {
  const { categoryId } = await params;

  const res = await query<GetCategoryQuery, GetCategoryQueryVariables>({
    query: GET_CATEGORY,
    variables: { id: categoryId, eventsLimit: 20, spacesLimit: 20 },
  });

  const category = res.data?.category;
  if (!category) notFound();

  const session = await auth.api
    .getSession({ headers: await headers() })
    .catch(() => null);
  if (session?.user) {
    recordImpression(session.user.id, categoryId, "category", "viewed");
  }

  return (
    <Page
      breadcrumbs={[
        { label: "Categories", href: "/categories" },
        { label: category.id },
      ]}
      header={
        <h1 className="text-5xl font-extrabold tracking-tight capitalize">
          {category.id}
        </h1>
      }
    >
      <CategoryContent
        categoryId={categoryId}
        events={category.recommendedEvents ?? []}
        spaces={category.recommendedSpaces ?? []}
        similar={category.recommendedCategories ?? []}
      />
    </Page>
  );
}
