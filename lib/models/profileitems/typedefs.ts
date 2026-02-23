
export const userItemTypeDefs = `#graphql
  """
  A single item on a user's profile — either a photo or a prompt answer.
  """
  type ProfileItem {
    id: ID!
    userId: ID!
    type: UserItemType!
    promptKey: String
    content: String!
    displayOrder: Int!
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  enum UserItemType {
    photo
    prompt
  }

  input AddUserItemInput {
    type: UserItemType!
    promptKey: String
    content: String!
    displayOrder: Int
  }

  input UpdateUserItemInput {
    content: String
    promptKey: String
  }

  extend type Query {
    userItems(userId: ID!): [ProfileItem!]!
  }

  extend type Mutation {
    addUserItem(input: AddUserItemInput!): ProfileItem!
    updateUserItem(itemId: ID!, input: UpdateUserItemInput!): ProfileItem!
    deleteUserItem(itemId: ID!): Boolean!
    reorderUserItems(itemIds: [ID!]!): [ProfileItem!]!
  }
`;
