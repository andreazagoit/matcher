/**
 * GraphQL Schema for Profiles
 */

export const profileTypeDefs = `#graphql
  type Profile {
    id: ID!
    userId: ID!
    updatedAt: String!
  }

  extend type Query {
    """
    Get the authenticated user's profile.
    """
    myProfile: Profile
  }
`;
