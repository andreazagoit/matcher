import {
  createUser,
  updateUser,
  deleteUser,
  getUserById,
  getAllUsers,
  findMatches,
} from "./operations";
import type { CreateUserInput, UpdateUserInput } from "./validator";

export const userResolvers = {
  Query: {
    user: async (_: unknown, { id }: { id: string }) => {
      return await getUserById(id);
    },

    users: async () => {
      return await getAllUsers();
    },

    me: async (_: unknown, __: unknown, context: { userId?: string }) => {
      const userId = context.userId;
      if (!userId) return null;
      return await getUserById(userId);
    },

    findMatches: async (
      _: unknown,
      { userId, limit = 10 }: { userId: string; limit?: number }
    ) => {
      return await findMatches(userId, { limit });
    },
  },

  Mutation: {
    createUser: async (
      _: unknown,
      { input }: { input: CreateUserInput }
    ) => {
      return await createUser(input);
    },

    updateUser: async (
      _: unknown,
      { id, input }: { id: string; input: UpdateUserInput }
    ) => {
      return await updateUser(id, input);
    },

    deleteUser: async (_: unknown, { id }: { id: string }) => {
      return await deleteUser(id);
    },
  },
};
