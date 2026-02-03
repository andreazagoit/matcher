import {
  createUser,
  updateUser,
  deleteUser,
  getUserById,
  getAllUsers,
} from "./operations";
import {
  findMatches,
  getProfileByUserId,
} from "@/lib/models/profiles/operations";
import type { CreateUserInput, UpdateUserInput } from "./validator";
import type { User } from "./schema";

interface MatchOptions {
  limit?: number;
  gender?: ("man" | "woman" | "non_binary")[];
  minAge?: number;
  maxAge?: number;
}

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
      { userId, options }: { userId: string; options?: MatchOptions }
    ) => {
      const matches = await findMatches(userId, {
        limit: options?.limit ?? 10,
        gender: options?.gender,
        minAge: options?.minAge,
        maxAge: options?.maxAge,
      });
      
      // Ritorna solo gli utenti
      return matches.map((match) => match.user);
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

  // Field resolvers
  User: {
    profile: async (user: User) => {
      return await getProfileByUserId(user.id);
    },
  },
};
