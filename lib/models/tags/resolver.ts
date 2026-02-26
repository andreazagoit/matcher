import { getTagCategories, getAllTags, createTag } from "./operations";

export const tagResolvers = {
  Query: {
    tagCategories: async () => await getTagCategories(),
    allTags: async () => await getAllTags(),
  },
  Mutation: {
    createTag: async (_: any, { name, category }: { name: string; category: string }) => {
      // Returns the normalized name of the created tag string
      return await createTag(name, category);
    },
  },
};
