export const profileItemTypeDefs = `#graphql
  """
  A single item on a user's profile — either a photo or a prompt answer.
  """
  type ProfileItem {
    id: ID!
    userId: ID!
    type: ProfileItemType!
    promptKey: String
    content: String!
    displayOrder: Int!
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  enum ProfileItemType {
    photo
    prompt
  }

  input AddProfileItemInput {
    type: ProfileItemType!
    promptKey: String
    content: String!
    displayOrder: Int
  }

  input UpdateProfileItemInput {
    content: String
    promptKey: String
  }

  extend type Query {
    profileItems(userId: ID!): [ProfileItem!]!
  }

  extend type Mutation {
    addProfileItem(input: AddProfileItemInput!): ProfileItem!
    updateProfileItem(itemId: ID!, input: UpdateProfileItemInput!): ProfileItem!
    deleteProfileItem(itemId: ID!): Boolean!
    reorderProfileItems(itemIds: [ID!]!): [ProfileItem!]!
  }
`;
