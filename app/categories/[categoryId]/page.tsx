import { notFound } from "next/navigation";
import { headers } from "next/headers";
import { query } from "@/lib/graphql/apollo-client";
import { GET_CATEGORY } from "@/lib/models/categories/gql";
import { Page } from "@/components/page";
import { auth } from "@/lib/auth";
import { recordImpression } from "@/lib/models/impressions/operations";
import { CategoryContent } from "./category-content";

interface CategoryEvent {
  id: string;
  title: string;
  description?: string | null;
  location?: string | null;
  startsAt: string;
  endsAt?: string | null;
  attendeeCount: number;
  maxAttendees?: number | null;
  categories: string[];
  spaceId: string;
  price?: number | null;
  currency?: string | null;
  isPaid: boolean;
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
        <h1 className="text-5xl font-extrabold tracking-tight capitalize">
          {category.id}
        </h1>
      }
    >
      <CategoryContent
        categoryId={categoryId}
        events={events}
        spaces={spaces}
        similar={similar}
      />
    </Page>
  );
}
