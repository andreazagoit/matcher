export const notificationTypeDefs = `#graphql
  enum NotificationType {
    new_match
    match_mutual
    new_message
    space_joined
    event_reminder
    event_rsvp
    generic
  }

  type Notification {
    id: ID!
    type: NotificationType!
    text: String!
    image: String
    href: String
    read: Boolean!
    createdAt: DateTime!
  }

  type NotificationsResult {
    items: [Notification!]!
    unreadCount: Int!
  }

  extend type User {
    notifications(limit: Int, offset: Int): NotificationsResult!
  }

  extend type Mutation {
    markNotificationRead(id: ID!): Notification
    markAllNotificationsRead: Boolean!
    deleteNotification(id: ID!): Boolean!
  }
`;
