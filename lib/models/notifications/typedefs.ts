export const notificationTypeDefs = `#graphql
  type Notification {
    id: ID!
    type: String!
    text: String!
    image: String
    href: String
    read: Boolean!
    createdAt: DateTime!
  }

  extend type Query {
    notifications(limit: Int, offset: Int): [Notification!]!
    unreadNotificationsCount: Int!
  }

  extend type Mutation {
    markNotificationRead(id: ID!): Notification
    markAllNotificationsRead: Boolean!
    deleteNotification(id: ID!): Boolean!
  }
`;
