import { getUserItems } from "./operations";

export const userItemResolvers = {
  User: {
    userItems: async (parent: { id: string }) => {
      return getUserItems(parent.id);
    },
  },
};
