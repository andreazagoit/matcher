export const spaceTypeDefs = `#graphql
  enum SpaceVisibility {
    public
    private
    hidden
  }

  enum JoinPolicy {
    open
    apply
    invite_only
  }

  type Space {
    id: ID!
    name: String!
    slug: String!
    description: String
    cover: String!
    images: [String!]!
    categories: [String!]!
    visibility: SpaceVisibility!
    joinPolicy: JoinPolicy!
    createdAt: DateTime!
    membersCount: Int
    stripeAccountEnabled: Boolean!
    """Events belonging to this space, ordered by startsAt."""
    events(limit: Int, offset: Int): EventConnection!
    """Upcoming events from other spaces with similar embeddings (AI-recommended)."""
    recommendedEvents(limit: Int): [Event!]!
  }

  input CreateSpaceInput {
    name: String!
    slug: String!
    description: String
    cover: String!
    images: [String!]
    categories: [String!]
    visibility: SpaceVisibility
    joinPolicy: JoinPolicy
  }

  input UpdateSpaceInput {
    name: String
    description: String
    cover: String
    images: [String!]
    categories: [String!]
    visibility: SpaceVisibility
    joinPolicy: JoinPolicy
  }

  extend type Query {
    space(id: ID, slug: String): Space
    spaces(limit: Int, offset: Int): SpaceConnection!
    mySpaces: [Space!]!
  }

  extend type Mutation {
    createSpace(input: CreateSpaceInput!): Space!
    updateSpace(id: ID!, input: UpdateSpaceInput!): Space!
    deleteSpace(id: ID!): Boolean!
  }
`;
