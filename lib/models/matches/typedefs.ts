/**
 * GraphQL types for matching engine v2.
 */

export const matchTypeDefs = `#graphql

  type Match {
    user: MatchUser!
    score: Float!
    distanceKm: Float
    sharedTags: [String!]!
    sharedSpaceIds: [String!]!
    sharedEventIds: [String!]!
  }

  type MatchUser {
    id: ID!
    username: String!
    name: String!
    image: String
    gender: String
    birthdate: String!
    userItems: [UserItem!]!
  }

  type ProfileStatus {
    hasProfile: Boolean!
    updatedAt: DateTime
  }

  # ── Queries ────────────────────────────────────────────────────

  extend type Query {
    """
    Find compatible matches for the authenticated user.
    Uses tag overlap, shared spaces/events, proximity, and behavioral similarity.
    """
    findMatches(maxDistance: Float! = 50, limit: Int, offset: Int, gender: [String!], minAge: Int, maxAge: Int): [Match!]!

    """
    Get the authenticated user's profile status.
    """
    profileStatus: ProfileStatus!
  }
`;
