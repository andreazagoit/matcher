import { getCategories, getCategoryById, createCategory } from "./operations";

export const categoryResolvers = {
  Query: {
    categories: async () => await getCategories(),
    category: async (_: unknown, { id }: { id: string }) => await getCategoryById(id),
  },
  Mutation: {
    createCategory: async (_: unknown, { name }: { name: string }) => {
      const id = await createCategory(name);
      return { id, name: id };
    },
  },
};
