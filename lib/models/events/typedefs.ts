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
    maxAttendees: Int
    categories: [String!]!

    createdBy: User!
    createdAt: DateTime!
    updatedAt: DateTime!
    attendees: [EventAttendee!]!
    attendeeCount: Int!
    """Status of the currently authenticated user for this event (null if not authenticated or not RSVP'd)"""
    myAttendeeStatus: AttendeeStatus
    """The space this event belongs to"""
    space: Space
    """Ticket price in cents (null = free event)"""
    price: Int
    """ISO 4217 currency code, e.g. 'eur'"""
    currency: String
    """True when the event requires purchasing a ticket"""
    isPaid: Boolean!
    """Payment status for the currently authenticated user (null if free event or no purchase)"""
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
    lat: Float
    lon: Float
    startsAt: String!
    endsAt: String
    maxAttendees: Int
    categories: [String!]
    price: Int
    currency: String
  }

  input UpdateEventInput {
    title: String
    description: String
    location: String
    lat: Float
    lon: Float
    startsAt: String
    endsAt: String
    maxAttendees: Int
    categories: [String!]

    price: Int
    currency: String
  }

  extend type Query {
    event(id: ID!): Event
    myUpcomingEvents: [Event!]!
  }

  extend type Mutation {
    """
    Create a new event in a space.
    """
    createEvent(input: CreateEventInput!): Event!

    """
    Update an existing event.
    """
    updateEvent(id: ID!, input: UpdateEventInput!): Event!

    """
    Respond to an event (going, interested).
    """
    respondToEvent(eventId: ID!, status: AttendeeStatus!): EventAttendee!

    """
    Mark an event as completed. Attendees with status 'going' become 'attended'.
    """
    markEventCompleted(eventId: ID!): Event!
  }
`;
