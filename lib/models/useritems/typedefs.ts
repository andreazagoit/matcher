
export const userItemTypeDefs = `#graphql
  """
  A single item on a user's profile — either a photo or a prompt answer.
  """
  type UserItem {
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
    userItems(userId: ID!): [UserItem!]!
  }

  extend type Mutation {
    addUserItem(input: AddUserItemInput!): UserItem!
    updateUserItem(itemId: ID!, input: UpdateUserItemInput!): UserItem!
    deleteUserItem(itemId: ID!): Boolean!
    reorderUserItems(itemIds: [ID!]!): [UserItem!]!
  }
`;
