import { userResolvers } from "../models/users/resolver";

export const resolvers = {
  Query: {
    ...userResolvers.Query,
  },
  Mutation: {
    ...userResolvers.Mutation,
  },
};

