/**
 * GraphQL Schema for Events
 */

export const eventTypeDefs = `#graphql
  enum PaymentStatus {
    pending
    paid
    refunded
  }

  type Event {
    id: ID!
    spaceId: ID!
    title: String!
    description: String
    location: String
    coordinates: Coordinates
    startsAt: DateTime!
    endsAt: DateTime
    cover: String!
    images: [String!]!
    categories: [String!]!
    maxAttendees: Int
    """Price in cents (e.g. 1000 = €10.00). Null for free events."""
    price: Int
    currency: String
    createdBy: User!
    createdAt: DateTime!
    updatedAt: DateTime!
    attendees: [EventAttendee!]!
    attendeeCount: Int!
    """Viewer's RSVP status. Null if not authenticated or not responded."""
    myAttendeeStatus: AttendeeStatus
    """The space this event belongs to."""
    space: Space
    """True when the event requires purchasing a ticket."""
    isPaid: Boolean!
    """Viewer's payment status. Null if free event or no purchase."""
    myPaymentStatus: PaymentStatus
    """Events with similar embeddings (AI-recommended). Excludes this event."""
    recommendedEvents(limit: Int): [Event!]!
  }

  type EventAttendee {
    id: ID!
    eventId: ID!
    userId: ID!
    user: User
    status: AttendeeStatus!
    registeredAt: DateTime!
    attendedAt: DateTime
    paymentStatus: PaymentStatus
  }

  enum AttendeeStatus {
    going
    interested
    attended
  }

  input CreateEventInput {
    spaceId: ID!
    title: String!
    description: String
    location: String
    coordinates: CoordinatesInput
    startsAt: DateTime!
    endsAt: DateTime
    cover: String!
    images: [String!]
    categories: [String!]
    maxAttendees: Int
    """Price in cents (e.g. 1000 = €10.00). Omit or 0 for free events."""
    price: Int
    currency: String
  }

  input UpdateEventInput {
    title: String
    description: String
    location: String
    coordinates: CoordinatesInput
    startsAt: DateTime
    endsAt: DateTime
    cover: String
    images: [String!]
    categories: [String!]
    maxAttendees: Int
    """Price in cents (e.g. 1000 = €10.00). Omit or 0 for free events."""
    price: Int
    currency: String
  }

  extend type Query {
    event(id: ID!): Event
    events(limit: Int, offset: Int): EventConnection!
    myUpcomingEvents: [Event!]!
  }

  extend type Mutation {
    """Create a new event in a space."""
    createEvent(input: CreateEventInput!): Event!

    """Update an existing event."""
    updateEvent(id: ID!, input: UpdateEventInput!): Event!

    """Respond to an event (going, interested)."""
    respondToEvent(eventId: ID!, status: AttendeeStatus!): EventAttendee!

    """Mark an event as completed. Attendees with status 'going' become 'attended'."""
    markEventCompleted(eventId: ID!): Event!
  }
`;
