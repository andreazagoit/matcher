import { getTagCategories, getAllTags } from "./operations";

export const tagResolvers = {
  Query: {
    tagCategories: () => getTagCategories(),
    allTags: () => getAllTags(),
  },
};
