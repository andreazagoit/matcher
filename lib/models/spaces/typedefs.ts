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

  enum SpaceType {
    free
    tiered
  }

  type Space {
    id: ID!
    name: String!
    slug: String!
    description: String
    image: String
    categories: [String!]!
    visibility: SpaceVisibility!
    joinPolicy: JoinPolicy!
    createdAt: DateTime!
    isActive: Boolean
    membersCount: Int
    type: SpaceType
    stripeAccountEnabled: Boolean!
    """Events belonging to this space, ordered by startsAt."""
    events(limit: Int, offset: Int): [Event!]!
    """Upcoming events from other spaces with similar embeddings (AI-recommended)."""
    recommendedEvents(limit: Int): [Event!]!
  }

  input CreateSpaceInput {
    name: String!
    slug: String
    description: String
    visibility: SpaceVisibility
    joinPolicy: JoinPolicy
    categories: [String!]
  }

  input UpdateSpaceInput {
    name: String
    description: String
    visibility: SpaceVisibility
    joinPolicy: JoinPolicy
    image: String
    categories: [String!]
  }

  extend type Query {
    space(id: ID, slug: String): Space
    spaces: [Space!]!
    mySpaces: [Space!]!
  }

  extend type Mutation {
    createSpace(input: CreateSpaceInput!): Space!
    updateSpace(id: ID!, input: UpdateSpaceInput!): Space!
    deleteSpace(id: ID!): Boolean!
  }
`;
